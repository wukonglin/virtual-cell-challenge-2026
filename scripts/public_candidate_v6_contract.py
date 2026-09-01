#!/usr/bin/env python3
"""Create and verify immutable public-validation candidate contracts.

The ``plan`` command authenticates every generator-side input without opening
sealed treated truth, including the exact generator code and STATE auxiliary
files.  The ``verify`` command checks that a generated H5AD and its report
exactly implement the planned context, residual, alpha, checkpoint, code/model
provenance, and count-decoding configuration.  Scoring jobs must require a
successful verification receipt before they are allowed to read scorer-only
truth.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd

from generate_state_direct_counts import validate_selection_manifest
from infer_state_effect_prior import (
    CONTROL_LABEL,
    EXPECTED_SUPPORT_GENES,
    describe_file,
    read_single_column,
    require,
    sha256_file,
)


LEGACY_SPEC_SCHEMA = "vcc-public-candidate-spec-v1"
SPEC_SCHEMA = "vcc-public-candidate-spec-v2"
VERIFICATION_SCHEMA = "vcc-public-candidate-verification-v1"
PANEL_SCHEMA = "vcc-public-validation-manifest-v1"
RESIDUAL_SCHEMA = "vcc-public-state-residual-v1"
DATA_SCHEMA = "vcc-public-validation-data-v1"
PREDICTION_SCHEMA = "vcc-native-state-anchor-counts-v1"
TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
CONTEXT_PREFIX = {"HepG2": "hepg2", "Jurkat": "jurkat"}
DEFAULT_TARGET_REMAINING_FRACTION = 0.20
APPROVED_TARGET_REMAINING_FRACTIONS = (
    DEFAULT_TARGET_REMAINING_FRACTION,
    0.40,
)
DEFAULT_STATE_EFFECT_WEIGHT = 1.00
APPROVED_STATE_EFFECT_WEIGHTS = (
    0.50,
    0.75,
    DEFAULT_STATE_EFFECT_WEIGHT,
)
P4_STATE_GAMMA_TAGS = {
    0.50: "p4_g050_p0_a010",
    0.75: "p4_g075_p0_a010",
    1.00: "p4_g100_p0_a010",
}

STRICT_PROVENANCE_CONTRACT = "vcc-public-generator-provenance-v2"
STRICT_SPEC_INPUT_KEYS = (
    "controls_h5ad",
    "panel_manifest",
    "panel_csv",
    "residual_npz",
    "residual_json",
    "checkpoint",
    "selection_json",
    "support_genes",
    "generator_script",
    "generate_state_direct_counts_helper",
    "infer_state_effect_prior_helper",
    "perturbation_map",
    "var_dims",
    "state_config",
)
STRICT_GENERATOR_REPORT_INPUT_MAP = (
    ("script", "generator_script"),
    ("perturbation_map", "perturbation_map"),
    ("var_dims", "var_dims"),
)
OPTIONAL_STRICT_INPUT_KEYS = frozenset({"state_config"})

LOCKED_GENERATOR_CONFIGURATION: dict[str, Any] = {
    "seed": 20260901,
    "cells_per_group": 400,
    "model_chunk_size": 128,
    "state_effect_weight": DEFAULT_STATE_EFFECT_WEIGHT,
    "state_effect_clip": 0.60,
    "state_effect_bounding": "tanh",
    "combined_effect_clip": 0.65,
    "target_policy": "force",
    "max_genes_per_cell": 5900,
    "integerization": "largest-remainder",
}


def authenticate_state_source(
    source_dir: Path,
    *,
    require_active_isolation: bool = False,
    expected_isolation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind the clean tracked Git tree that supplies the STATE runtime."""

    source_dir = source_dir.resolve()
    require(source_dir.is_dir(), f"Missing STATE source directory: {source_dir}")

    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(source_dir), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )
        require(
            completed.returncode == 0,
            f"Unable to authenticate STATE source Git tree: {completed.stderr.strip()}",
        )
        return completed.stdout.strip()

    repository_root = Path(git("rev-parse", "--show-toplevel")).resolve()
    require(repository_root == source_dir, "STATE source path is not its Git repository root")
    commit = git("rev-parse", "HEAD")
    tree = git("rev-parse", "HEAD^{tree}")
    tracked_status = git("status", "--porcelain", "--untracked-files=no")
    require(not tracked_status, "STATE source has tracked worktree changes")
    untracked = git("ls-files", "--others", "--exclude-standard", "--", "src").splitlines()
    ignored = git(
        "ls-files", "--others", "--ignored", "--exclude-standard", "--", "src"
    ).splitlines()
    unsafe_runtime_files = sorted(path for path in {*untracked, *ignored} if path)
    require(
        not unsafe_runtime_files,
        f"STATE source has untracked or ignored import-affecting files: {unsafe_runtime_files}",
    )
    if expected_isolation is None:
        source_path = (source_dir / "src").resolve()
        python_path_entries = [
            Path(item).resolve()
            for item in os.environ.get("PYTHONPATH", "").split(os.pathsep)
            if item
        ]
        no_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        require(
            not require_active_isolation
            or (
                python_path_entries
                and python_path_entries[0] == source_path
                and no_bytecode
            ),
            "STATE runtime isolation environment is not active",
        )
        isolation = {
            "mode": "authenticated-clean-worktree-no-bytecode-v1",
            "pythonpath_first": str(source_path),
            "python_dont_write_bytecode": True,
        }
    else:
        isolation = expected_isolation
    return {
        "path": str(source_dir),
        "repository_commit": commit,
        "repository_tree": tree,
        "tracked_worktree_clean": True,
        "untracked_and_ignored_runtime_files_absent": True,
        "runtime_isolation": isolation,
    }


def _strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "Contract output must end in .json")
    require(not path.exists(), f"Refusing to overwrite contract: {path}")
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


def _descriptor_matches(
    observed: dict[str, Any], expected: dict[str, Any], label: str
) -> None:
    require(
        Path(str(observed.get("path", ""))).resolve()
        == Path(str(expected.get("path", ""))).resolve(),
        f"{label} path differs from the contract",
    )
    require(
        int(observed.get("size_bytes", -1)) == int(expected.get("size_bytes", -2)),
        f"{label} size differs from the contract",
    )
    require(
        str(observed.get("sha256", "")) == str(expected.get("sha256", "missing")),
        f"{label} SHA-256 differs from the contract",
    )


def _authenticate_descriptor(
    path: Path, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label}: {path}")
    observed = describe_file(path)
    _descriptor_matches(observed, expected, label)
    return observed


def _validate_strict_spec_provenance(spec: dict[str, Any]) -> dict[str, Any]:
    """Require the complete v2 generator-input provenance declaration."""

    require(spec.get("schema") == SPEC_SCHEMA, "Bad strict candidate spec schema")
    contract = spec.get("provenance_contract", {})
    require(
        contract
        == {
            "schema": STRICT_PROVENANCE_CONTRACT,
            "mode": "strict",
            "generator_report_provenance_authenticated": True,
            "optional_state_config_bound": True,
        },
        "Strict generator provenance contract is absent or malformed",
    )
    inputs = spec.get("inputs", {})
    require(isinstance(inputs, dict), "Strict candidate spec inputs are absent")
    require(
        set(inputs) == set(STRICT_SPEC_INPUT_KEYS),
        "Strict candidate spec input fields are incomplete or unexpected",
    )
    for name in STRICT_SPEC_INPUT_KEYS:
        descriptor = inputs[name]
        if name in OPTIONAL_STRICT_INPUT_KEYS and descriptor is None:
            continue
        require(isinstance(descriptor, dict), f"Strict input descriptor is invalid: {name}")
        require(
            isinstance(descriptor.get("path"), str)
            and int(descriptor.get("size_bytes", -1)) >= 0
            and isinstance(descriptor.get("sha256"), str)
            and len(descriptor["sha256"]) == 64,
            f"Strict input descriptor is incomplete: {name}",
        )
    state_source = spec.get("state_source")
    require(
        isinstance(state_source, dict)
        and set(state_source)
        == {
            "path",
            "repository_commit",
            "repository_tree",
            "tracked_worktree_clean",
            "untracked_and_ignored_runtime_files_absent",
            "runtime_isolation",
        }
        and state_source.get("tracked_worktree_clean") is True
        and state_source.get("untracked_and_ignored_runtime_files_absent") is True
        and state_source.get("runtime_isolation", {}).get("mode")
        == "authenticated-clean-worktree-no-bytecode-v1"
        and state_source.get("runtime_isolation", {}).get("python_dont_write_bytecode") is True
        and isinstance(state_source.get("path"), str)
        and len(str(state_source.get("repository_commit", ""))) == 40
        and len(str(state_source.get("repository_tree", ""))) == 40,
        "Strict STATE runtime source contract is absent or malformed",
    )
    return inputs


def validate_output_tag(value: str) -> str:
    require(bool(TAG_PATTERN.fullmatch(value)), "Unsafe output tag")
    return value


def validate_target_remaining_fraction(value: float) -> float:
    """Return a canonical, pre-registered P3 target-force setting."""

    require(math.isfinite(value), "Target remaining fraction must be finite")
    for approved in APPROVED_TARGET_REMAINING_FRACTIONS:
        if value == approved:
            return approved
    approved_values = ", ".join(
        f"{item:.2f}" for item in APPROVED_TARGET_REMAINING_FRACTIONS
    )
    raise RuntimeError(
        "Target remaining fraction is outside the pre-registered P3 set "
        f"({approved_values})"
    )


def validate_state_effect_weight(value: float) -> float:
    """Return a canonical, pre-registered STATE-anchor amplitude."""

    require(math.isfinite(value), "STATE effect weight must be finite")
    for approved in APPROVED_STATE_EFFECT_WEIGHTS:
        if value == approved:
            return approved
    approved_values = ", ".join(
        f"{item:.2f}" for item in APPROVED_STATE_EFFECT_WEIGHTS
    )
    raise RuntimeError(
        "STATE effect weight is outside the pre-registered V6 set "
        f"({approved_values})"
    )


def validate_p4_state_gamma_arm(
    *,
    context: str,
    output_tag: str,
    state_effect_weight: float,
    residual_alpha: float,
    target_remaining_fraction: float,
) -> None:
    """Bind every non-default gamma arm to the exact HepG2 P4 contract."""

    is_p4_tag = output_tag.startswith("p4_g")
    is_nondefault_gamma = state_effect_weight != DEFAULT_STATE_EFFECT_WEIGHT
    if not is_p4_tag and not is_nondefault_gamma:
        return
    require(context == "HepG2", "P4 STATE-gamma arms are restricted to HepG2")
    require(
        output_tag == P4_STATE_GAMMA_TAGS[state_effect_weight],
        "P4 output tag does not match the STATE effect weight",
    )
    require(residual_alpha == 0.10, "P4 STATE-gamma arms require residual alpha 0.10")
    require(
        target_remaining_fraction == 0.20,
        "P4 STATE-gamma arms require target remaining fraction 0.20",
    )


def validate_panel(
    manifest_path: Path, csv_path: Path
) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(payload.get("schema") == PANEL_SCHEMA, "Bad panel manifest schema")
    require(
        payload.get("selection_contract", {}).get("effect_values_accessed") is False,
        "Panel selection accessed effect values",
    )
    recorded = payload.get("provenance", {}).get("output_csv", {})
    require(isinstance(recorded, dict), "Panel manifest lacks CSV provenance")
    _authenticate_descriptor(csv_path, recorded, "panel CSV")
    frame = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    require(
        {"target_gene", "validation_route"}.issubset(frame.columns),
        "Panel CSV lacks required columns",
    )
    targets = frame["target_gene"].astype(str).tolist()
    routes = frame["validation_route"].astype(str).to_numpy()
    require(len(targets) == 300 and len(targets) == len(set(targets)), "Bad panel targets")
    require(set(routes) == {"direct", "held_target"}, "Bad panel routes")
    direct = routes == "direct"
    require(int(direct.sum()) == 267 and int((~direct).sum()) == 33, "Bad route counts")
    return targets, direct, payload


def validate_controls(path: Path, context: str) -> tuple[list[str], int]:
    data = ad.read_h5ad(path, backed="r")
    try:
        contract = data.uns.get("public_validation", {})
        require(
            contract.get("schema") == DATA_SCHEMA
            and contract.get("role") == "controls-only-generator-input"
            and contract.get("sealed_treated_profiles_present") is False,
            "Controls lack the generator-only firewall role",
        )
        require({"context", "target_gene"}.issubset(data.obs.columns), "Controls obs is incomplete")
        require(set(data.obs["context"].astype(str)) == {context}, "Controls context mismatch")
        require(
            set(data.obs["target_gene"].astype(str)) == {CONTROL_LABEL},
            "Controls contain treated cells",
        )
        require(data.n_obs >= 400, "Controls contain fewer than 400 cells")
        genes = data.var_names.astype(str).tolist()
        require(genes and len(genes) == len(set(genes)), "Controls gene axis is invalid")
        return genes, int(data.n_obs)
    finally:
        data.file.close()


def validate_residual(
    npz_path: Path,
    json_path: Path,
    *,
    context: str,
    targets: list[str],
    direct: np.ndarray,
    native_genes: list[str],
) -> dict[str, Any]:
    report = json.loads(json_path.read_text(encoding="utf-8"))
    require(report.get("schema") == RESIDUAL_SCHEMA, "Bad residual sidecar schema")
    require(report.get("context") == context, "Residual sidecar context mismatch")
    contract = report.get("contract", {})
    require(contract.get("held_target_rows_are_zero") is True, "Residual held rows are unsafe")
    require(
        contract.get("recipient_treated_profiles_used") is False,
        "Residual used recipient treated profiles",
    )
    require(
        contract.get("native_non_state_genes_are_immutable") is True,
        "Residual does not protect non-STATE genes",
    )
    require(
        float(report.get("configuration", {}).get("shared_response_weight", math.nan))
        == 0.0,
        "Residual includes a shared response",
    )
    expected_npz = report.get("provenance", {}).get("output_npz", {})
    require(isinstance(expected_npz, dict), "Residual sidecar lacks NPZ provenance")
    _authenticate_descriptor(npz_path, expected_npz, "residual NPZ")

    with np.load(npz_path, allow_pickle=False) as archive:
        required = {
            "effects",
            "target_names",
            "gene_names",
            "contexts",
            "direct_mask",
            "fallback_mask",
        }
        require(required.issubset(archive.files), "Residual NPZ lacks required arrays")
        residual_targets = _strings(archive["target_names"])
        residual_genes = _strings(archive["gene_names"])
        residual_contexts = _strings(archive["contexts"])
        effects = np.asarray(archive["effects"], dtype=np.float32)
        residual_direct = np.asarray(archive["direct_mask"]).reshape(-1).astype(bool)
        residual_fallback = np.asarray(archive["fallback_mask"]).reshape(-1).astype(bool)
        require(residual_targets == targets, "Residual target order differs from panel")
        require(residual_genes == native_genes, "Residual gene order differs from controls")
        require(residual_contexts == [context], "Residual NPZ context mismatch")
        require(effects.shape == (1, 300, len(native_genes)), "Bad residual shape")
        require(np.isfinite(effects).all(), "Residual contains non-finite values")
        require(np.array_equal(residual_direct, direct), "Residual direct mask differs from panel")
        require(np.array_equal(residual_fallback, ~direct), "Residual fallback mask differs from panel")
        require(not np.any(effects[:, residual_fallback]), "Fallback residual rows are nonzero")
        require(np.count_nonzero(effects[:, residual_direct]) > 0, "Direct residual rows are all zero")
    return report


def build_spec(args: argparse.Namespace) -> dict[str, Any]:
    require(args.context in CONTEXT_PREFIX, "Unsupported public-validation context")
    tag = validate_output_tag(args.output_tag)
    require(
        math.isfinite(args.residual_alpha) and args.residual_alpha >= 0,
        "Alpha must be finite and nonnegative",
    )
    state_effect_weight = validate_state_effect_weight(args.state_effect_weight)
    target_remaining_fraction = validate_target_remaining_fraction(
        args.target_remaining_fraction
    )
    validate_p4_state_gamma_arm(
        context=args.context,
        output_tag=tag,
        state_effect_weight=state_effect_weight,
        residual_alpha=float(args.residual_alpha),
        target_remaining_fraction=target_remaining_fraction,
    )
    paths = (
        args.controls_h5ad,
        args.panel_manifest,
        args.panel_csv,
        args.residual_npz,
        args.residual_json,
        args.checkpoint,
        args.selection_json,
        args.support_genes,
        args.generator_script,
        args.generate_state_direct_counts_helper,
        args.infer_state_effect_prior_helper,
        args.perturbation_map,
        args.var_dims,
    )
    for path in paths:
        require(path.is_file(), f"Missing candidate input: {path}")
    generator_directory = args.generator_script.resolve().parent
    require(
        args.generate_state_direct_counts_helper.resolve()
        == generator_directory / "generate_state_direct_counts.py"
        and args.infer_state_effect_prior_helper.resolve()
        == generator_directory / "infer_state_effect_prior.py",
        "Executable helper paths do not match the generator import directory",
    )
    if args.state_config is not None:
        require(args.state_config.is_file(), f"Missing candidate input: {args.state_config}")
    state_source = authenticate_state_source(
        args.state_source_dir, require_active_isolation=True
    )

    targets, direct, _ = validate_panel(args.panel_manifest, args.panel_csv)
    native_genes, control_cells = validate_controls(args.controls_h5ad, args.context)
    residual_report = validate_residual(
        args.residual_npz,
        args.residual_json,
        context=args.context,
        targets=targets,
        direct=direct,
        native_genes=native_genes,
    )
    support_genes = read_single_column(args.support_genes)
    require(len(support_genes) == EXPECTED_SUPPORT_GENES, "Unexpected STATE support size")
    checkpoint_sha256 = sha256_file(args.checkpoint)
    selection = validate_selection_manifest(
        args.selection_json, args.checkpoint, checkpoint_sha256
    )

    configuration = dict(LOCKED_GENERATOR_CONFIGURATION)
    configuration["context"] = args.context
    configuration["state_effect_weight"] = state_effect_weight
    configuration["residual_alpha"] = float(args.residual_alpha)
    configuration["target_remaining_fraction"] = target_remaining_fraction
    return {
        "schema": SPEC_SCHEMA,
        "output_tag": tag,
        "context": args.context,
        "context_prefix": CONTEXT_PREFIX[args.context],
        "configuration": configuration,
        "axes": {
            "targets": len(targets),
            "direct_targets": int(direct.sum()),
            "held_targets": int((~direct).sum()),
            "native_genes": len(native_genes),
            "control_cells": control_cells,
            "state_support_genes": len(support_genes),
        },
        "inputs": {
            "controls_h5ad": describe_file(args.controls_h5ad),
            "panel_manifest": describe_file(args.panel_manifest),
            "panel_csv": describe_file(args.panel_csv),
            "residual_npz": describe_file(args.residual_npz),
            "residual_json": describe_file(args.residual_json),
            "checkpoint": {
                **describe_file(args.checkpoint, hash_file=False),
                "sha256": checkpoint_sha256,
            },
            "selection_json": describe_file(args.selection_json),
            "support_genes": describe_file(args.support_genes),
            "generator_script": describe_file(args.generator_script),
            "generate_state_direct_counts_helper": describe_file(
                args.generate_state_direct_counts_helper
            ),
            "infer_state_effect_prior_helper": describe_file(
                args.infer_state_effect_prior_helper
            ),
            "perturbation_map": describe_file(args.perturbation_map),
            "var_dims": describe_file(args.var_dims),
            "state_config": (
                describe_file(args.state_config) if args.state_config is not None else None
            ),
        },
        "checkpoint_selection": {
            "global_step": int(selection["selected"]["global_step"]),
            "val_loss": float(selection["selected"]["val_loss"]),
        },
        "state_source": state_source,
        "residual_contract": {
            "schema": residual_report["schema"],
            "held_target_rows_are_zero": True,
            "recipient_treated_profiles_used": False,
            "shared_response_weight": float(
                residual_report.get("configuration", {}).get("shared_response_weight", math.nan)
            ),
        },
        "firewall": {
            "generator_inputs_include_sealed_truth": False,
            "controls_role": "controls-only-generator-input",
            "treated_profiles_read_while_planning": False,
        },
        "provenance_contract": {
            "schema": STRICT_PROVENANCE_CONTRACT,
            "mode": "strict",
            "generator_report_provenance_authenticated": True,
            "optional_state_config_bound": True,
        },
    }


def validate_generation_report(
    report: dict[str, Any],
    spec: dict[str, Any],
    *,
    allow_legacy_v1_read_only: bool = False,
) -> None:
    schema = spec.get("schema")
    require(
        schema == SPEC_SCHEMA
        or (schema == LEGACY_SPEC_SCHEMA and allow_legacy_v1_read_only),
        "Bad candidate spec schema; legacy v1 requires explicit read-only validation",
    )
    strict_inputs = _validate_strict_spec_provenance(spec) if schema == SPEC_SCHEMA else None
    require(report.get("schema") == PREDICTION_SCHEMA, "Bad generation report schema")
    require(report.get("artifact_type") == "sealed_public_validation_prediction", "Bad artifact type")
    require(report.get("full_frozen_panel_contract") is True, "Prediction is not full-panel")
    firewall = report.get("data_firewall", {})
    require(firewall.get("sealed_treated_profiles_read") is False, "Generator read treated truth")
    require(firewall.get("truth_inputs_read") == [], "Generator reports truth inputs")
    configuration = report.get("configuration", {})
    for key, expected in spec["configuration"].items():
        observed = configuration.get(key)
        if isinstance(expected, float):
            require(
                math.isfinite(float(observed)) and float(observed) == expected,
                f"Generation parameter differs from spec: {key}",
            )
        else:
            require(observed == expected, f"Generation parameter differs from spec: {key}")
    validation = report.get("validation", {})
    expected_shape = [
        spec["axes"]["targets"] * spec["configuration"]["cells_per_group"],
        spec["axes"]["native_genes"],
    ]
    require(validation.get("shape") == expected_shape, "Generated shape differs from spec")
    require(validation.get("groups") == spec["axes"]["targets"], "Generated group count differs")
    require(validation.get("failed_checks") == [], "Generated candidate failed checks")
    scientific = report.get("scientific_qc", {})
    require(scientific.get("all_libraries_exact") is True, "Generated libraries drifted")
    require(
        scientific.get("all_native_non_support_counts_exact") is True,
        "Generated non-STATE counts drifted",
    )
    require(scientific.get("target_knockdown_failures") == [], "Target clamp failed")
    provenance = report.get("provenance", {})
    for report_key, spec_key in (
        ("controls_h5ad", "controls_h5ad"),
        ("panel_manifest", "panel_manifest"),
        ("panel_csv", "panel_csv"),
        ("support_gene_axis", "support_genes"),
        ("checkpoint", "checkpoint"),
        ("checkpoint_selection", "selection_json"),
    ):
        observed = provenance.get(report_key)
        require(isinstance(observed, dict), f"Generation report lacks provenance: {report_key}")
        _descriptor_matches(observed, spec["inputs"][spec_key], report_key)

    if strict_inputs is not None:
        for report_key, spec_key in STRICT_GENERATOR_REPORT_INPUT_MAP:
            observed = provenance.get(report_key)
            require(
                isinstance(observed, dict),
                f"Generation report lacks strict provenance: {report_key}",
            )
            _descriptor_matches(observed, strict_inputs[spec_key], report_key)
        expected_state_config = strict_inputs["state_config"]
        observed_state_config = provenance.get("state_config")
        if expected_state_config is None:
            require(
                observed_state_config is None,
                "Generation report used an unbound STATE model config",
            )
        else:
            require(
                isinstance(observed_state_config, dict),
                "Generation report lacks bound STATE model config provenance",
            )
            _descriptor_matches(
                observed_state_config,
                expected_state_config,
                "state_config",
            )

    residual = report.get("residual", {})
    require(isinstance(residual, dict), "Generation report lacks residual QC")
    residual_alpha = float(spec["configuration"]["residual_alpha"])
    require(
        math.isfinite(float(residual.get("alpha", math.nan)))
        and float(residual["alpha"]) == residual_alpha,
        "Generation residual alpha differs from spec",
    )
    if residual_alpha == 0:
        require(residual.get("enabled") is False, "Zero-alpha residual is enabled")
        require(residual.get("artifact_read") is False, "Zero-alpha residual was read")
        require(
            Path(str(residual.get("declared_path", ""))).resolve()
            == Path(spec["inputs"]["residual_npz"]["path"]).resolve(),
            "Zero-alpha residual declaration differs from spec",
        )
        require(
            provenance.get("residual_artifact") is None,
            "Zero-alpha generator unexpectedly reports a read residual artifact",
        )
    else:
        require(residual.get("enabled") is True, "Positive-alpha residual is disabled")
        require(residual.get("artifact_read") is True, "Positive-alpha residual was not read")
        observed = provenance.get("residual_artifact")
        require(isinstance(observed, dict), "Generation report lacks residual provenance")
        _descriptor_matches(observed, spec["inputs"]["residual_npz"], "residual_artifact")


def verify_generation(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.spec, args.prediction_h5ad, args.generation_json):
        require(path.is_file(), f"Missing verification input: {path}")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    report = json.loads(args.generation_json.read_text(encoding="utf-8"))
    schema = spec.get("schema")
    allow_legacy = bool(getattr(args, "allow_legacy_v1_read_only", False))
    require(
        not allow_legacy or schema == LEGACY_SPEC_SCHEMA,
        "Legacy read-only mode is only valid for a v1 candidate spec",
    )
    validate_output_tag(str(spec.get("output_tag", "")))
    validate_generation_report(
        report,
        spec,
        allow_legacy_v1_read_only=allow_legacy,
    )
    authenticated_strict_inputs: dict[str, dict[str, Any] | None] = {}
    if schema == SPEC_SCHEMA:
        strict_inputs = _validate_strict_spec_provenance(spec)
        authenticated_state_source = authenticate_state_source(
            Path(str(spec.get("state_source", {}).get("path", ""))),
            require_active_isolation=True,
        )
        require(
            authenticated_state_source == spec.get("state_source"),
            "STATE runtime source differs from strict spec",
        )
        require(
            report.get("provenance", {}).get("state_source_commit")
            == authenticated_state_source["repository_commit"],
            "Generation report STATE source commit differs from strict spec",
        )
        for name in (
            "generator_script",
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
            "perturbation_map",
            "var_dims",
            "state_config",
        ):
            descriptor = strict_inputs[name]
            if descriptor is None:
                authenticated_strict_inputs[name] = None
                continue
            authenticated_strict_inputs[name] = _authenticate_descriptor(
                Path(str(descriptor["path"])), descriptor, f"strict input {name}"
            )
    expected_output = report.get("provenance", {}).get("output_h5ad", {})
    require(isinstance(expected_output, dict), "Generation report lacks output provenance")
    _authenticate_descriptor(args.prediction_h5ad, expected_output, "prediction H5AD")

    prediction = ad.read_h5ad(args.prediction_h5ad, backed="r")
    try:
        expected_shape = (
            spec["axes"]["targets"] * spec["configuration"]["cells_per_group"],
            spec["axes"]["native_genes"],
        )
        require(prediction.shape == expected_shape, "Prediction H5AD shape differs from spec")
        require({"context", "target_gene"}.issubset(prediction.obs.columns), "Prediction obs is incomplete")
        require(
            set(prediction.obs["context"].astype(str)) == {spec["context"]},
            "Prediction H5AD context differs from spec",
        )
        labels = prediction.obs["target_gene"].astype(str)
        require(CONTROL_LABEL not in set(labels), "Prediction H5AD contains controls")
        counts = labels.value_counts()
        require(len(counts) == spec["axes"]["targets"], "Prediction target count differs")
        require(
            set(counts.astype(int)) == {spec["configuration"]["cells_per_group"]},
            "Prediction cells per target differ from spec",
        )
    finally:
        prediction.file.close()

    return {
        "schema": VERIFICATION_SCHEMA,
        "status": "passed",
        "output_tag": spec["output_tag"],
        "context": spec["context"],
        "configuration": spec["configuration"],
        "provenance": {
            "spec": describe_file(args.spec),
            "generation_json": describe_file(args.generation_json),
            "prediction_h5ad": describe_file(args.prediction_h5ad),
            "authenticated_strict_inputs": authenticated_strict_inputs,
            "authenticated_state_source": (
                authenticated_state_source if schema == SPEC_SCHEMA else None
            ),
        },
        "checks": {
            "generator_inputs_authenticated": True,
            "generator_did_not_read_treated_truth": True,
            "generation_matches_locked_configuration": True,
            "prediction_hash_matches_report": True,
            "prediction_shape_and_groups_match_spec": True,
            "exact_library_and_non_state_invariants_passed": True,
            "strict_generator_provenance_authenticated": schema == SPEC_SCHEMA,
        },
        "validation_mode": (
            "strict-v2" if schema == SPEC_SCHEMA else "legacy-v1-read-only"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="Authenticate inputs and write an immutable spec.")
    plan.add_argument("--context", choices=tuple(CONTEXT_PREFIX), required=True)
    plan.add_argument("--output-tag", required=True)
    plan.add_argument(
        "--residual-alpha",
        type=float,
        required=True,
        help=(
            "Finite nonnegative residual multiplier; zero defines the STATE-only "
            "P3 arm while retaining authenticated residual declarations."
        ),
    )
    plan.add_argument(
        "--state-effect-weight",
        type=float,
        default=DEFAULT_STATE_EFFECT_WEIGHT,
        choices=APPROVED_STATE_EFFECT_WEIGHTS,
        help=(
            "Raw paired-STATE residual amplitude applied before tanh bounding. "
            "Only the pre-registered V6 values 0.50, 0.75, and 1.00 are accepted."
        ),
    )
    plan.add_argument(
        "--target-remaining-fraction",
        type=float,
        default=DEFAULT_TARGET_REMAINING_FRACTION,
        choices=APPROVED_TARGET_REMAINING_FRACTIONS,
        help=(
            "Forced CRISPRi target fold. Only the locked 0.20 default and the "
            "pre-registered P3 0.40 ablation are accepted."
        ),
    )
    plan.add_argument("--controls-h5ad", type=Path, required=True)
    plan.add_argument("--panel-manifest", type=Path, required=True)
    plan.add_argument("--panel-csv", type=Path, required=True)
    plan.add_argument("--residual-npz", type=Path, required=True)
    plan.add_argument("--residual-json", type=Path, required=True)
    plan.add_argument("--checkpoint", type=Path, required=True)
    plan.add_argument("--selection-json", type=Path, required=True)
    plan.add_argument("--support-genes", type=Path, required=True)
    plan.add_argument("--generator-script", type=Path, required=True)
    plan.add_argument(
        "--generate-state-direct-counts-helper", type=Path, required=True
    )
    plan.add_argument("--infer-state-effect-prior-helper", type=Path, required=True)
    plan.add_argument(
        "--state-source-dir",
        type=Path,
        required=True,
        help="Clean tracked Git repository supplying the STATE runtime.",
    )
    plan.add_argument("--perturbation-map", type=Path, required=True)
    plan.add_argument("--var-dims", type=Path, required=True)
    plan.add_argument(
        "--state-config",
        type=Path,
        default=None,
        help=(
            "Optional STATE model config. When supplied, its descriptor must "
            "match generation-report provenance; when omitted, the report must "
            "also declare no config."
        ),
    )
    plan.add_argument("--output-json", type=Path, required=True)

    verify = subparsers.add_parser("verify", help="Verify a completed candidate before scoring.")
    verify.add_argument("--spec", type=Path, required=True)
    verify.add_argument("--prediction-h5ad", type=Path, required=True)
    verify.add_argument("--generation-json", type=Path, required=True)
    verify.add_argument("--output-json", type=Path, default=None)
    verify.add_argument(
        "--allow-legacy-v1-read-only",
        action="store_true",
        help=(
            "Audit an already completed v1 candidate without writing a new "
            "verification receipt. New candidates must use strict v2 specs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "plan":
        payload = build_spec(args)
    else:
        payload = verify_generation(args)
    if args.command == "verify" and args.allow_legacy_v1_read_only:
        require(args.output_json is None, "Legacy v1 validation is read-only")
    else:
        require(args.output_json is not None, "Strict verification requires --output-json")
        _atomic_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
