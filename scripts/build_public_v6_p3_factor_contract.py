#!/usr/bin/env python3
"""Build or authenticate the immutable four-arm P3 factor contract.

The ``build`` command binds the legacy V5.1 HepG2 A/B arms to strict-v2 V6
C/D arms before factorial model selection.  It re-authenticates the V5.1
scoring receipt, all four full generation reports and predictions, every
shared generator input, the strict specifications and verification receipts,
and the frozen routing manifest.  It permits only the registered two-by-two
factor levels: target remaining fraction 0.20/0.40 and residual alpha
0.00/0.10.

The command opens prediction H5ADs only in backed mode.  It requires exact
var axes, obs annotations, and source-control pairing across all four arms,
then re-runs the C/D held-target count-identity gate in bounded CSR chunks.  It
does not consume expression truth.  The ``validate`` command re-hashes every
artifact bound by a completed factor receipt and is intended for use by the
factorial analyzer before selection.
"""

from __future__ import annotations

import argparse
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

from compare_public_p3_count_identity import (
    _authenticate_file,
    _descriptor_identity,
    stream_compare_counts,
    validate_axes_and_pairing,
)
from infer_state_effect_prior import describe_file, require
from public_candidate_v6_contract import (
    PREDICTION_SCHEMA,
    SPEC_SCHEMA,
    STRICT_PROVENANCE_CONTRACT,
    STRICT_SPEC_INPUT_KEYS,
    VERIFICATION_SCHEMA,
    _validate_strict_spec_provenance,
    authenticate_state_source,
    validate_generation_report,
    validate_output_tag,
)
from validate_public_v51_scoring_receipts import validate_scoring_receipt


SCHEMA = "vcc-public-v6-p3-factor-contract-v1"
CONTEXT = "HepG2"
ARM_KEYS = ("A", "B", "C", "D")
FACTOR_LEVELS = {
    "A": {"target_remaining_fraction": 0.20, "residual_alpha": 0.00},
    "B": {"target_remaining_fraction": 0.20, "residual_alpha": 0.10},
    "C": {"target_remaining_fraction": 0.40, "residual_alpha": 0.00},
    "D": {"target_remaining_fraction": 0.40, "residual_alpha": 0.10},
}
ARM_IDENTITIES = {
    "A": "anchor",
    "B": "v51_alpha010",
    "C": "p3_tf040_state_a000",
    "D": "p3_tf040_p0_a010",
}
FACTOR_NAMES = frozenset({"target_remaining_fraction", "residual_alpha"})
V51_SHARED_INPUT_MAP = {
    "controls_h5ad": "controls_h5ad",
    "panel_manifest": "panel_manifest",
    "panel_csv": "panel_csv",
    "checkpoint": "checkpoint",
    "selection_json": "checkpoint_selection",
    "support_genes": "support_gene_axis",
    "generator_script": "script",
    "perturbation_map": "perturbation_map",
    "var_dims": "var_dims",
    "state_config": "state_config",
}
STABLE_GENERATION_PROVENANCE = (
    "state_source_commit",
)
PRIMARY_ARTIFACT_KEYS = (
    "builder_script",
    "manifest",
    "v51_scoring_receipt",
    "A_generation_json",
    "A_prediction_h5ad",
    "B_generation_json",
    "B_prediction_h5ad",
    "C_spec",
    "C_generation_verification",
    "C_generation_json",
    "C_prediction_h5ad",
    "D_spec",
    "D_generation_verification",
    "D_generation_json",
    "D_prediction_h5ad",
)


@dataclass(frozen=True)
class FactorArm:
    """One fully authenticated generation arm."""

    key: str
    generation_path: Path
    prediction_path: Path
    generation: dict[str, Any]
    configuration: dict[str, Any]
    descriptors: dict[str, dict[str, Any]]
    spec_path: Path | None = None
    verification_path: Path | None = None
    spec: dict[str, Any] | None = None


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "Factor-contract output must end in .json")
    require(not path.exists(), f"Refusing to overwrite factor contract: {path}")
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


def _load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid {label}: {path}") from error
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(
        _descriptor_identity(left) == _descriptor_identity(right),
        f"Shared generator descriptor differs: {label}",
    )


def _validate_generation_common(report: dict[str, Any], *, arm: str) -> None:
    require(report.get("schema") == PREDICTION_SCHEMA, f"Arm {arm}: bad generation schema")
    require(
        report.get("artifact_type") == "sealed_public_validation_prediction"
        and report.get("full_frozen_panel_contract") is True,
        f"Arm {arm}: generation is not a sealed full-panel prediction",
    )
    firewall = report.get("data_firewall", {})
    require(
        firewall.get("sealed_treated_profiles_read") is False
        and firewall.get("truth_inputs_read") == [],
        f"Arm {arm}: generation reports treated-truth access",
    )
    validation = report.get("validation", {})
    validation_checks = validation.get("checks")
    require(
        validation.get("failed_checks") == []
        and isinstance(validation_checks, dict)
        and bool(validation_checks)
        and all(value is True for value in validation_checks.values()),
        f"Arm {arm}: generation validation did not pass",
    )
    scientific = report.get("scientific_qc", {})
    require(
        scientific.get("all_libraries_exact") is True
        and scientific.get("all_native_non_support_counts_exact") is True
        and scientific.get("target_knockdown_failures") == [],
        f"Arm {arm}: generation scientific invariants failed",
    )
    configuration = report.get("configuration")
    require(
        isinstance(configuration, dict) and configuration.get("context") == CONTEXT,
        f"Arm {arm}: generation configuration is incomplete",
    )


def authenticate_v51_arm(
    *,
    arm: str,
    generation_path: Path,
    prediction_path: Path,
    scoring_record: dict[str, Any],
    cache: dict[tuple[str, int, str], dict[str, Any]],
) -> FactorArm:
    """Bind one legacy generation report to its authenticated scoring prediction."""

    require(arm in ("A", "B"), "Legacy arms must be A or B")
    role = "anchor" if arm == "A" else "v51_alpha010"
    generation = _load_json(generation_path, f"arm {arm} generation report")
    _validate_generation_common(generation, arm=arm)
    recorded_prediction = scoring_record["predictions"][role]
    observed_prediction = _authenticate_file(
        prediction_path,
        recorded_prediction,
        f"arm {arm} scoring-bound prediction",
        cache,
    )
    output_descriptor = generation.get("provenance", {}).get("output_h5ad", {})
    _same_descriptor(output_descriptor, recorded_prediction, f"arm {arm} prediction")
    _authenticate_file(
        prediction_path, output_descriptor, f"arm {arm} generation output", cache
    )

    residual = generation.get("residual", {})
    expected_alpha = FACTOR_LEVELS[arm]["residual_alpha"]
    require(float(residual.get("alpha", math.nan)) == expected_alpha, f"Arm {arm}: residual alpha mismatch")
    if expected_alpha == 0:
        require(
            residual.get("enabled") is False
            and residual.get("artifact_read") is False
            and generation.get("provenance", {}).get("residual_artifact") is None,
            "Arm A: zero-alpha path read a residual artifact",
        )
    else:
        require(
            residual.get("enabled") is True
            and residual.get("artifact_read") is True
            and isinstance(generation.get("provenance", {}).get("residual_artifact"), dict),
            "Arm B: positive-alpha residual provenance is absent",
        )

    return FactorArm(
        key=arm,
        generation_path=generation_path.resolve(),
        prediction_path=prediction_path.resolve(),
        generation=generation,
        configuration=dict(generation["configuration"]),
        descriptors={
            "generation_json": describe_file(generation_path),
            "prediction_h5ad": observed_prediction,
        },
    )


def authenticate_strict_arm(
    *,
    arm: str,
    spec_path: Path,
    verification_path: Path,
    generation_path: Path,
    prediction_path: Path,
    cache: dict[tuple[str, int, str], dict[str, Any]],
) -> FactorArm:
    """Authenticate one strict-v2 arm and every declared input descriptor."""

    require(arm in ("C", "D"), "Strict arms must be C or D")
    spec = _load_json(spec_path, f"arm {arm} strict specification")
    verification = _load_json(verification_path, f"arm {arm} generation verification")
    generation = _load_json(generation_path, f"arm {arm} generation report")
    require(spec.get("schema") == SPEC_SCHEMA, f"Arm {arm}: strict-v2 spec is required")
    validate_output_tag(str(spec.get("output_tag", "")))
    require(
        spec.get("output_tag") == ARM_IDENTITIES[arm],
        f"Arm {arm}: strict output tag differs from the registered identity",
    )
    strict_inputs = _validate_strict_spec_provenance(spec)
    require(
        verification.get("schema") == VERIFICATION_SCHEMA
        and verification.get("status") == "passed"
        and verification.get("validation_mode") == "strict-v2",
        f"Arm {arm}: strict generation verification did not pass",
    )
    checks = verification.get("checks", {})
    require(
        isinstance(checks, dict)
        and checks.get("strict_generator_provenance_authenticated") is True
        and all(value is True for value in checks.values()),
        f"Arm {arm}: strict verification checks are incomplete",
    )
    require(
        verification.get("output_tag") == spec.get("output_tag")
        and verification.get("context") == CONTEXT
        and verification.get("configuration") == spec.get("configuration"),
        f"Arm {arm}: verification identity differs from strict spec",
    )
    verification_provenance = verification.get("provenance", {})
    verified_strict_inputs = verification_provenance.get("authenticated_strict_inputs")
    require(
        isinstance(verified_strict_inputs, dict)
        and set(verified_strict_inputs)
        == {
            "generator_script",
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
            "perturbation_map",
            "var_dims",
            "state_config",
        },
        f"Arm {arm}: post-generation strict input authentication is incomplete",
    )
    observed_spec = _authenticate_file(
        spec_path, verification_provenance.get("spec", {}), f"arm {arm} spec", cache
    )
    observed_generation = _authenticate_file(
        generation_path,
        verification_provenance.get("generation_json", {}),
        f"arm {arm} generation report",
        cache,
    )
    observed_prediction = _authenticate_file(
        prediction_path,
        verification_provenance.get("prediction_h5ad", {}),
        f"arm {arm} prediction",
        cache,
    )
    _validate_generation_common(generation, arm=arm)
    validate_generation_report(generation, spec)
    _same_descriptor(
        generation.get("provenance", {}).get("output_h5ad", {}),
        verification_provenance.get("prediction_h5ad", {}),
        f"arm {arm} generation output",
    )
    authenticated_inputs: dict[str, dict[str, Any]] = {}
    for name in STRICT_SPEC_INPUT_KEYS:
        descriptor = strict_inputs[name]
        if descriptor is None:
            continue
        input_path = Path(str(descriptor["path"])).resolve()
        authenticated_inputs[name] = _authenticate_file(
            input_path, descriptor, f"arm {arm} strict input {name}", cache
        )
        if name in verified_strict_inputs:
            _same_descriptor(
                verified_strict_inputs[name],
                descriptor,
                f"arm {arm} post-generation strict input {name}",
            )
    require(
        verified_strict_inputs.get("state_config") == strict_inputs["state_config"],
        f"Arm {arm}: post-generation STATE config binding differs",
    )

    return FactorArm(
        key=arm,
        generation_path=generation_path.resolve(),
        prediction_path=prediction_path.resolve(),
        generation=generation,
        configuration=dict(generation["configuration"]),
        descriptors={
            "spec": observed_spec,
            "generation_verification": describe_file(verification_path),
            "generation_json": observed_generation,
            "prediction_h5ad": observed_prediction,
            **{f"input_{name}": value for name, value in authenticated_inputs.items()},
        },
        spec_path=spec_path.resolve(),
        verification_path=verification_path.resolve(),
        spec=spec,
    )


def validate_registered_factor_grid(arms: dict[str, FactorArm]) -> dict[str, Any]:
    """Require a complete 2x2 design with no unregistered configuration drift."""

    require(set(arms) == set(ARM_KEYS), "Factor contract requires exactly arms A/B/C/D")
    configurations = {arm: arms[arm].configuration for arm in ARM_KEYS}
    key_sets = {frozenset(configuration) for configuration in configurations.values()}
    require(len(key_sets) == 1, "Four-arm generation configuration keys differ")
    keys = next(iter(key_sets))
    require(FACTOR_NAMES.issubset(keys), "Registered factor fields are absent")
    for arm, expected in FACTOR_LEVELS.items():
        for factor, level in expected.items():
            observed = float(configurations[arm].get(factor, math.nan))
            require(
                math.isfinite(observed) and observed == level,
                f"Arm {arm} has the wrong registered factor level: {factor}",
            )
    non_factor = sorted(keys - FACTOR_NAMES)
    for name in non_factor:
        require(
            len({json.dumps(configurations[arm][name], sort_keys=True) for arm in ARM_KEYS}) == 1,
            f"Unregistered generation configuration drift: {name}",
        )
    return {
        "levels": FACTOR_LEVELS,
        "arm_identities": ARM_IDENTITIES,
        "registered_factors": sorted(FACTOR_NAMES),
        "non_factor_configuration_fields_exact": non_factor,
        "complete_two_by_two_grid": True,
    }


def validate_shared_generation_identity(arms: dict[str, FactorArm]) -> dict[str, Any]:
    """Compare generator inputs and stable model provenance across schemas."""

    reference_report = arms["A"].generation
    for arm in ARM_KEYS[1:]:
        report = arms[arm].generation
        for field in ("axes", "checkpoint_selection", "model"):
            require(
                report.get(field) == reference_report.get(field),
                f"Arm {arm}: shared generation field differs: {field}",
            )
        for field in STABLE_GENERATION_PROVENANCE:
            require(
                report.get("provenance", {}).get(field)
                == reference_report.get("provenance", {}).get(field),
                f"Arm {arm}: stable provenance differs: {field}",
            )

    shared: dict[str, dict[str, Any]] = {}
    for normalized_name, v51_name in V51_SHARED_INPUT_MAP.items():
        descriptors = {
            "A": arms["A"].generation.get("provenance", {}).get(v51_name),
            "B": arms["B"].generation.get("provenance", {}).get(v51_name),
            "C": arms["C"].spec.get("inputs", {}).get(normalized_name),
            "D": arms["D"].spec.get("inputs", {}).get(normalized_name),
        }
        reference = descriptors["A"]
        require(isinstance(reference, dict), f"Missing shared generator input: {normalized_name}")
        for arm in ARM_KEYS[1:]:
            _same_descriptor(reference, descriptors[arm], f"{normalized_name}|A_vs_{arm}")
        shared[normalized_name] = {
            "path": str(Path(str(reference["path"])).resolve()),
            "size_bytes": int(reference["size_bytes"]),
            "sha256": str(reference["sha256"]),
        }

    residual_b = arms["B"].generation.get("provenance", {}).get("residual_artifact")
    residual_c = arms["C"].spec.get("inputs", {}).get("residual_npz")
    residual_d = arms["D"].spec.get("inputs", {}).get("residual_npz")
    _same_descriptor(residual_b, residual_c, "power-zero residual B_vs_C")
    _same_descriptor(residual_b, residual_d, "power-zero residual B_vs_D")
    _same_descriptor(
        residual_b,
        arms["D"].generation.get("provenance", {}).get("residual_artifact"),
        "power-zero residual B_vs_D-generation",
    )
    require(
        arms["A"].generation.get("provenance", {}).get("residual_artifact") is None
        and arms["C"].generation.get("provenance", {}).get("residual_artifact") is None,
        "Zero-alpha arms must not read the residual artifact",
    )
    residual_json_c = arms["C"].spec.get("inputs", {}).get("residual_json")
    residual_json_d = arms["D"].spec.get("inputs", {}).get("residual_json")
    _same_descriptor(residual_json_c, residual_json_d, "strict residual sidecar C_vs_D")
    strict_helper_inputs: dict[str, dict[str, Any]] = {}
    for name in (
        "generate_state_direct_counts_helper",
        "infer_state_effect_prior_helper",
    ):
        descriptor_c = arms["C"].spec.get("inputs", {}).get(name)
        descriptor_d = arms["D"].spec.get("inputs", {}).get(name)
        _same_descriptor(descriptor_c, descriptor_d, f"strict executable helper C_vs_D: {name}")
        strict_helper_inputs[name] = {
            "path": str(Path(str(descriptor_c["path"])).resolve()),
            "size_bytes": int(descriptor_c["size_bytes"]),
            "sha256": str(descriptor_c["sha256"]),
        }
    strict_state_source = arms["C"].spec.get("state_source")
    require(
        strict_state_source == arms["D"].spec.get("state_source"),
        "Strict C/D STATE runtime source contracts differ",
    )
    require(
        strict_state_source.get("repository_commit")
        == arms["C"].generation.get("provenance", {}).get("state_source_commit")
        == arms["D"].generation.get("provenance", {}).get("state_source_commit"),
        "Strict C/D STATE runtime commit differs from generation provenance",
    )
    return {
        "shared_generator_inputs": shared,
        "residual_npz": {
            "path": str(Path(str(residual_b["path"])).resolve()),
            "size_bytes": int(residual_b["size_bytes"]),
            "sha256": str(residual_b["sha256"]),
        },
        "strict_only_inputs": {
            "residual_json": {
                "path": str(Path(str(residual_json_c["path"])).resolve()),
                "size_bytes": int(residual_json_c["size_bytes"]),
                "sha256": str(residual_json_c["sha256"]),
            },
            **strict_helper_inputs,
        },
        "strict_state_source": strict_state_source,
        "stable_generation_provenance": {
            field: reference_report.get("provenance", {}).get(field)
            for field in STABLE_GENERATION_PROVENANCE
        },
        "noncausal_main_repository_commits_by_arm": {
            arm: arms[arm].generation.get("provenance", {}).get("repository_commit")
            for arm in ARM_KEYS
        },
    }


def _load_manifest(path: Path, arms: dict[str, FactorArm]) -> tuple[list[str], np.ndarray, dict[str, Any]]:
    descriptor = arms["C"].spec.get("inputs", {}).get("panel_csv")
    _same_descriptor(
        descriptor,
        arms["D"].spec.get("inputs", {}).get("panel_csv"),
        "strict panel CSV",
    )
    require(path.resolve() == Path(str(descriptor["path"])).resolve(), "Manifest path differs from specs")
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    require(
        {"target_gene", "validation_route"}.issubset(frame.columns),
        "Manifest lacks target_gene or validation_route",
    )
    targets = frame["target_gene"].astype(str).tolist()
    routes = frame["validation_route"].astype(str).to_numpy()
    require(len(targets) == len(set(targets)), "Manifest target axis contains duplicates")
    require(set(routes) == {"direct", "held_target"}, "Manifest routes are invalid")
    direct = routes == "direct"
    axes = arms["C"].spec["axes"]
    require(
        len(targets) == int(axes["targets"])
        and int(direct.sum()) == int(axes["direct_targets"])
        and int((~direct).sum()) == int(axes["held_targets"]),
        "Manifest route counts differ from strict specs",
    )
    return targets, direct, describe_file(path)


def _validate_all_axes(
    arms: dict[str, FactorArm], targets: list[str], direct: np.ndarray, chunk_rows: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    data = {
        arm: ad.read_h5ad(record.prediction_path, backed="r")
        for arm, record in arms.items()
    }
    try:
        cells = int(arms["A"].configuration["cells_per_group"])
        pair_checks = {
            f"A_vs_{arm}": validate_axes_and_pairing(
                data["A"], data[arm], context=CONTEXT, targets=targets, cells_per_group=cells
            )
            for arm in ("B", "C", "D")
        }
        counts = stream_compare_counts(
            data["C"],
            data["D"],
            direct_mask=direct,
            cells_per_group=cells,
            chunk_rows=chunk_rows,
        )
    finally:
        for candidate in data.values():
            candidate.file.close()
    held = counts["held_target"]
    require(
        held["exact_count_identity"]
        and held["differing_rows"] == 0
        and held["differing_entries"] == 0,
        "C/D held-target counts differ; factorial contract is invalid",
    )
    return pair_checks, counts


def build_factor_contract(args: argparse.Namespace) -> dict[str, Any]:
    """Authenticate four arms and return a passing immutable factor contract."""

    require(args.chunk_rows > 0, "chunk-rows must be positive")
    scoring = validate_scoring_receipt(args.v51_scoring_receipt, CONTEXT)
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    arms = {
        "A": authenticate_v51_arm(
            arm="A",
            generation_path=args.arm_a_generation,
            prediction_path=args.arm_a_prediction,
            scoring_record=scoring,
            cache=cache,
        ),
        "B": authenticate_v51_arm(
            arm="B",
            generation_path=args.arm_b_generation,
            prediction_path=args.arm_b_prediction,
            scoring_record=scoring,
            cache=cache,
        ),
        "C": authenticate_strict_arm(
            arm="C",
            spec_path=args.arm_c_spec,
            verification_path=args.arm_c_verification,
            generation_path=args.arm_c_generation,
            prediction_path=args.arm_c_prediction,
            cache=cache,
        ),
        "D": authenticate_strict_arm(
            arm="D",
            spec_path=args.arm_d_spec,
            verification_path=args.arm_d_verification,
            generation_path=args.arm_d_generation,
            prediction_path=args.arm_d_prediction,
            cache=cache,
        ),
    }
    factor_grid = validate_registered_factor_grid(arms)
    shared_identity = validate_shared_generation_identity(arms)
    targets, direct, manifest_descriptor = _load_manifest(args.manifest, arms)
    axes, count_comparison = _validate_all_axes(arms, targets, direct, args.chunk_rows)

    primary_artifacts = {
        "builder_script": describe_file(Path(__file__)),
        "manifest": manifest_descriptor,
        "v51_scoring_receipt": describe_file(args.v51_scoring_receipt),
        "A_generation_json": arms["A"].descriptors["generation_json"],
        "A_prediction_h5ad": arms["A"].descriptors["prediction_h5ad"],
        "B_generation_json": arms["B"].descriptors["generation_json"],
        "B_prediction_h5ad": arms["B"].descriptors["prediction_h5ad"],
        "C_spec": arms["C"].descriptors["spec"],
        "C_generation_verification": arms["C"].descriptors["generation_verification"],
        "C_generation_json": arms["C"].descriptors["generation_json"],
        "C_prediction_h5ad": arms["C"].descriptors["prediction_h5ad"],
        "D_spec": arms["D"].descriptors["spec"],
        "D_generation_verification": arms["D"].descriptors["generation_verification"],
        "D_generation_json": arms["D"].descriptors["generation_json"],
        "D_prediction_h5ad": arms["D"].descriptors["prediction_h5ad"],
    }
    return {
        "schema": SCHEMA,
        "status": "passed",
        "context": CONTEXT,
        "factor_grid": factor_grid,
        "shared_generation_identity": shared_identity,
        "axes_and_source_control_pairing": {
            "all_four_arms_exact": True,
            "pair_checks": axes,
        },
        "cd_count_identity": {
            "held_target_hard_gate_passed": True,
            "comparison": count_comparison,
            "streaming_chunk_rows": int(args.chunk_rows),
        },
        "v51_scoring_identity": {
            key: scoring[key]
            for key in (
                "config_digest",
                "cell_eval2_version",
                "pdex_version",
                "anchor_semantic_identity",
                "source_fingerprint",
            )
        },
        "contract": {
            "strict_v2_required_for_c_and_d": True,
            "only_registered_factor_differences": True,
            "all_shared_generator_inputs_identical": True,
            "strict_cd_executable_helper_descriptors_identical": True,
            "all_axes_and_source_control_pairing_identical": True,
            "cd_held_target_counts_identical": True,
            "factor_builder_did_not_consume_treated_expression_values": True,
        },
        "provenance": {
            "primary_artifacts": primary_artifacts,
            "shared_generator_inputs": shared_identity["shared_generator_inputs"],
            "residual_npz": shared_identity["residual_npz"],
            "strict_only_inputs": shared_identity["strict_only_inputs"],
            "strict_state_source": shared_identity["strict_state_source"],
            "noncausal_main_repository_commits_by_arm": shared_identity[
                "noncausal_main_repository_commits_by_arm"
            ],
            "legacy_ab_helper_provenance": {
                "helper_descriptors_present_in_historical_reports": False,
                "compatibility_mode": "explicit-read-only-pre-strict-v2",
                "strict_cd_helpers_must_not_be_attributed_retroactively_to_ab": True,
            },
        },
    }


def authenticate_factor_contract_receipt(
    receipt_path: Path,
    *,
    expected_manifest: Path | None = None,
    expected_v51_scoring_receipt: Path | None = None,
    expected_c_spec: Path | None = None,
    expected_d_spec: Path | None = None,
) -> dict[str, Any]:
    """Re-hash and validate a factor receipt before factorial selection."""

    payload = _load_json(receipt_path, "P3 factor-contract receipt")
    require(
        payload.get("schema") == SCHEMA
        and payload.get("status") == "passed"
        and payload.get("context") == CONTEXT,
        "P3 factor-contract receipt did not pass",
    )
    contract = payload.get("contract", {})
    require(
        isinstance(contract, dict) and bool(contract) and all(value is True for value in contract.values()),
        "P3 factor-contract guarantees are incomplete",
    )
    require(payload.get("factor_grid", {}).get("levels") == FACTOR_LEVELS, "Factor levels differ")
    require(
        payload.get("factor_grid", {}).get("arm_identities") == ARM_IDENTITIES,
        "Factor arm identities differ",
    )
    require(
        payload.get("factor_grid", {}).get("complete_two_by_two_grid") is True,
        "Factor grid is incomplete",
    )
    held = payload.get("cd_count_identity", {}).get("comparison", {}).get("held_target", {})
    require(
        payload.get("cd_count_identity", {}).get("held_target_hard_gate_passed") is True
        and held.get("exact_count_identity") is True
        and held.get("differing_rows") == 0
        and held.get("differing_entries") == 0,
        "C/D held-target count identity is not established",
    )

    provenance = payload.get("provenance", {})
    recorded_main_commits = provenance.get("noncausal_main_repository_commits_by_arm")
    require(
        isinstance(recorded_main_commits, dict)
        and set(recorded_main_commits) == set(ARM_KEYS),
        "Per-arm noncausal main repository commits are absent",
    )
    require(
        provenance.get("legacy_ab_helper_provenance")
        == {
            "helper_descriptors_present_in_historical_reports": False,
            "compatibility_mode": "explicit-read-only-pre-strict-v2",
            "strict_cd_helpers_must_not_be_attributed_retroactively_to_ab": True,
        },
        "Legacy A/B helper provenance boundary is not explicit",
    )
    primary = provenance.get("primary_artifacts", {})
    require(set(primary) == set(PRIMARY_ARTIFACT_KEYS), "Primary factor artifacts are incomplete")
    expected = {
        "manifest": expected_manifest,
        "v51_scoring_receipt": expected_v51_scoring_receipt,
        "C_spec": expected_c_spec,
        "D_spec": expected_d_spec,
    }
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    authenticated_primary: dict[str, dict[str, Any]] = {}
    for name in PRIMARY_ARTIFACT_KEYS:
        descriptor = primary[name]
        recorded_path = Path(str(descriptor.get("path", ""))).resolve()
        if expected.get(name) is not None:
            require(recorded_path == expected[name].resolve(), f"Factor receipt path mismatch: {name}")
        authenticated_primary[name] = _authenticate_file(
            recorded_path, descriptor, f"factor receipt {name}", cache
        )
    shared_inputs = provenance.get("shared_generator_inputs", {})
    require(
        set(shared_inputs) == set(V51_SHARED_INPUT_MAP),
        "Shared generator input receipt fields are incomplete",
    )
    authenticated_shared = {
        name: _authenticate_file(
            Path(str(descriptor.get("path", ""))).resolve(),
            descriptor,
            f"factor receipt shared input {name}",
            cache,
        )
        for name, descriptor in shared_inputs.items()
    }
    residual = provenance.get("residual_npz", {})
    authenticated_residual = _authenticate_file(
        Path(str(residual.get("path", ""))).resolve(),
        residual,
        "factor receipt residual NPZ",
        cache,
    )
    strict_only = provenance.get("strict_only_inputs", {})
    require(
        set(strict_only)
        == {
            "residual_json",
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
        },
        "Strict-only factor inputs are incomplete",
    )
    authenticated_strict_only = {
        name: _authenticate_file(
            Path(str(descriptor.get("path", ""))).resolve(),
            descriptor,
            f"factor receipt strict-only input {name}",
            cache,
        )
        for name, descriptor in strict_only.items()
    }
    recorded_state_source = provenance.get("strict_state_source")
    require(isinstance(recorded_state_source, dict), "Strict STATE source receipt is absent")
    authenticated_state_source = authenticate_state_source(
        Path(str(recorded_state_source.get("path", ""))),
        expected_isolation=recorded_state_source.get("runtime_isolation"),
    )
    require(
        authenticated_state_source == recorded_state_source,
        "Strict STATE runtime source differs from factor receipt",
    )

    # Re-derive the registered factors and cross-artifact bindings from the
    # authenticated small JSON artifacts.  Selection must not trust receipt
    # booleans alone.
    a_generation = _load_json(
        Path(authenticated_primary["A_generation_json"]["path"]),
        "factor receipt arm A generation",
    )
    b_generation = _load_json(
        Path(authenticated_primary["B_generation_json"]["path"]),
        "factor receipt arm B generation",
    )
    c_spec = _load_json(Path(authenticated_primary["C_spec"]["path"]), "factor receipt arm C spec")
    d_spec = _load_json(Path(authenticated_primary["D_spec"]["path"]), "factor receipt arm D spec")
    c_verification = _load_json(
        Path(authenticated_primary["C_generation_verification"]["path"]),
        "factor receipt arm C verification",
    )
    d_verification = _load_json(
        Path(authenticated_primary["D_generation_verification"]["path"]),
        "factor receipt arm D verification",
    )
    c_generation = _load_json(
        Path(authenticated_primary["C_generation_json"]["path"]),
        "factor receipt arm C generation",
    )
    d_generation = _load_json(
        Path(authenticated_primary["D_generation_json"]["path"]),
        "factor receipt arm D generation",
    )
    reports = {
        "A": a_generation,
        "B": b_generation,
        "C": c_generation,
        "D": d_generation,
    }
    require(
        recorded_main_commits
        == {
            arm: reports[arm].get("provenance", {}).get("repository_commit")
            for arm in ARM_KEYS
        },
        "Per-arm noncausal main repository commits differ",
    )
    for arm, report in reports.items():
        _validate_generation_common(report, arm=arm)
        for factor, level in FACTOR_LEVELS[arm].items():
            require(
                float(report["configuration"].get(factor, math.nan)) == level,
                f"Factor receipt arm {arm} has the wrong {factor}",
            )
        _same_descriptor(
            report.get("provenance", {}).get("output_h5ad", {}),
            primary[f"{arm}_prediction_h5ad"],
            f"factor receipt arm {arm} output",
        )
    for arm, spec, verification in (
        ("C", c_spec, c_verification),
        ("D", d_spec, d_verification),
    ):
        verification_checks = verification.get("checks")
        _validate_strict_spec_provenance(spec)
        require(
            spec.get("schema") == SPEC_SCHEMA
            and spec.get("output_tag") == ARM_IDENTITIES[arm]
            and isinstance(spec.get("configuration"), dict),
            f"Factor receipt arm {arm} strict spec identity differs",
        )
        # Generation reports contain runtime-only fields in addition to the
        # strict planned configuration, so compare every planned key exactly.
        for name, value in spec.get("configuration", {}).items():
            require(
                reports[arm]["configuration"].get(name) == value,
                f"Factor receipt arm {arm} generation differs from strict spec: {name}",
            )
        require(
            verification.get("schema") == VERIFICATION_SCHEMA
            and verification.get("status") == "passed"
            and verification.get("validation_mode") == "strict-v2"
            and verification.get("output_tag") == ARM_IDENTITIES[arm]
            and verification.get("configuration") == spec.get("configuration")
            and isinstance(verification_checks, dict)
            and bool(verification_checks)
            and verification_checks.get("strict_generator_provenance_authenticated") is True
            and all(value is True for value in verification_checks.values()),
            f"Factor receipt arm {arm} strict verification differs",
        )
        for provenance_name, primary_name in (
            ("spec", f"{arm}_spec"),
            ("generation_json", f"{arm}_generation_json"),
            ("prediction_h5ad", f"{arm}_prediction_h5ad"),
        ):
            _same_descriptor(
                verification.get("provenance", {}).get(provenance_name, {}),
                primary[primary_name],
                f"factor receipt arm {arm} verification {provenance_name}",
            )
        verified_strict_inputs = verification.get("provenance", {}).get(
            "authenticated_strict_inputs"
        )
        require(
            spec.get("state_source") == recorded_state_source
            and verification.get("provenance", {}).get("authenticated_state_source")
            == recorded_state_source
            and reports[arm].get("provenance", {}).get("state_source_commit")
            == recorded_state_source["repository_commit"],
            f"Factor receipt arm {arm} STATE runtime source differs",
        )
        require(
            isinstance(verified_strict_inputs, dict)
            and set(verified_strict_inputs)
            == {
                "generator_script",
                "generate_state_direct_counts_helper",
                "infer_state_effect_prior_helper",
                "perturbation_map",
                "var_dims",
                "state_config",
            },
            f"Factor receipt arm {arm} post-generation strict inputs are incomplete",
        )
        for normalized_name, v51_name in V51_SHARED_INPUT_MAP.items():
            _same_descriptor(
                reports["A"].get("provenance", {}).get(v51_name),
                shared_inputs[normalized_name],
                f"factor receipt A shared input {normalized_name}",
            )
            _same_descriptor(
                reports["B"].get("provenance", {}).get(v51_name),
                shared_inputs[normalized_name],
                f"factor receipt B shared input {normalized_name}",
            )
            _same_descriptor(
                spec.get("inputs", {}).get(normalized_name),
                shared_inputs[normalized_name],
                f"factor receipt arm {arm} shared input {normalized_name}",
            )
        _same_descriptor(
            spec.get("inputs", {}).get("residual_npz"),
            residual,
            f"factor receipt arm {arm} residual NPZ",
        )
        _same_descriptor(
            spec.get("inputs", {}).get("residual_json"),
            strict_only["residual_json"],
            f"factor receipt arm {arm} residual sidecar",
        )
        for helper_name in (
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
        ):
            _same_descriptor(
                spec.get("inputs", {}).get(helper_name),
                strict_only[helper_name],
                f"factor receipt arm {arm} executable helper {helper_name}",
            )
            _same_descriptor(
                verified_strict_inputs.get(helper_name),
                strict_only[helper_name],
                f"factor receipt arm {arm} verified executable helper {helper_name}",
            )
    non_factor_reference = {
        key: value
        for key, value in reports["A"]["configuration"].items()
        if key not in FACTOR_NAMES
    }
    for arm in ARM_KEYS[1:]:
        observed = {
            key: value
            for key, value in reports[arm]["configuration"].items()
            if key not in FACTOR_NAMES
        }
        require(observed == non_factor_reference, f"Factor receipt arm {arm} has unregistered drift")

    v51_receipt = _load_json(
        Path(authenticated_primary["v51_scoring_receipt"]["path"]),
        "factor receipt V5.1 scoring receipt",
    )
    require(
        v51_receipt.get("schema") == "vcc-public-validation-scoring-receipt-v1"
        and v51_receipt.get("status") == "scoring_complete"
        and v51_receipt.get("context") == CONTEXT,
        "Factor receipt V5.1 scoring identity differs",
    )
    for role, arm in (("anchor", "A"), ("v51_alpha010", "B")):
        recorded = v51_receipt.get("predictions", {}).get(role, {})
        primary_prediction = primary[f"{arm}_prediction_h5ad"]
        require(
            Path(str(recorded.get("path", ""))).resolve()
            == Path(str(primary_prediction.get("path", ""))).resolve()
            and recorded.get("sha256") == primary_prediction.get("sha256"),
            f"Factor receipt V5.1 role differs: {role}",
        )
    return {
        "schema": SCHEMA,
        "status": "authenticated",
        "receipt": describe_file(receipt_path),
        "context": CONTEXT,
        "factor_levels": FACTOR_LEVELS,
        "primary_artifacts": authenticated_primary,
        "shared_generator_inputs": authenticated_shared,
        "residual_npz": authenticated_residual,
        "strict_only_inputs": authenticated_strict_only,
        "strict_state_source": authenticated_state_source,
        "cd_held_target_logical_csr_sha256": held["left_logical_csr_sha256"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build an immutable authenticated four-arm receipt.")
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--v51-scoring-receipt", type=Path, required=True)
    for arm in ("a", "b"):
        build.add_argument(f"--arm-{arm}-generation", type=Path, required=True)
        build.add_argument(f"--arm-{arm}-prediction", type=Path, required=True)
    for arm in ("c", "d"):
        build.add_argument(f"--arm-{arm}-spec", type=Path, required=True)
        build.add_argument(f"--arm-{arm}-verification", type=Path, required=True)
        build.add_argument(f"--arm-{arm}-generation", type=Path, required=True)
        build.add_argument(f"--arm-{arm}-prediction", type=Path, required=True)
    build.add_argument("--chunk-rows", type=int, default=400)
    build.add_argument("--output-json", type=Path, required=True)

    validate = commands.add_parser("validate", help="Re-authenticate a completed factor receipt.")
    validate.add_argument("--receipt", type=Path, required=True)
    validate.add_argument("--manifest", type=Path)
    validate.add_argument("--v51-scoring-receipt", type=Path)
    validate.add_argument("--arm-c-spec", type=Path)
    validate.add_argument("--arm-d-spec", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        payload = build_factor_contract(args)
        _atomic_json(args.output_json, payload)
    else:
        payload = authenticate_factor_contract_receipt(
            args.receipt,
            expected_manifest=args.manifest,
            expected_v51_scoring_receipt=args.v51_scoring_receipt,
            expected_c_spec=args.arm_c_spec,
            expected_d_spec=args.arm_d_spec,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
