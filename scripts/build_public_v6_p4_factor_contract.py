#!/usr/bin/env python3
"""Build or validate the strict three-arm P4 STATE-gamma contract.

All three candidates must use strict-v2 generation receipts. The contract
authenticates every generator input and prediction, then requires the arms to
differ only in ``state_effect_weight`` at the pre-registered levels 1.00,
0.75, and 0.50. It streams only generated count matrices to authenticate axes,
source-control pairing, and nonzero realized contrasts; it never reads truth.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np

from compare_public_p3_count_identity import (
    Candidate,
    _authenticate_file,
    _descriptor_identity,
    authenticate_candidate,
    stream_compare_counts,
    validate_axes_and_pairing,
)
from infer_state_effect_prior import describe_file, require
from public_candidate_v6_contract import SPEC_SCHEMA


SCHEMA = "vcc-public-v6-p4-factor-contract-v1"
CONTEXT = "HepG2"
FACTOR_NAME = "state_effect_weight"
ARM_DEFINITIONS = {
    "g100": {"output_tag": "p4_g100_p0_a010", "state_effect_weight": 1.00},
    "g075": {"output_tag": "p4_g075_p0_a010", "state_effect_weight": 0.75},
    "g050": {"output_tag": "p4_g050_p0_a010", "state_effect_weight": 0.50},
}
ARM_KEYS = tuple(ARM_DEFINITIONS)
PRIMARY_DESCRIPTOR_KEYS = (
    "spec",
    "generation_verified",
    "generation_json",
    "prediction_h5ad",
)
EXACT_SPEC_FIELDS = (
    "schema",
    "context",
    "context_prefix",
    "axes",
    "checkpoint_selection",
    "residual_contract",
    "firewall",
    "provenance_contract",
    "state_source",
)
PINNED_V51_INPUT_SHA256 = {
    "checkpoint": "e407e51855792d0dc85483a977f63497d9ac9a6893db360d840c7d9ce556780d",
    "controls_h5ad": "1a6ee4bc241c59abf6659205e40a96ca6f3af954500d10667a03a0cb8b0eee45",
    "generate_state_direct_counts_helper": "4b2e141904a5cb293af173ed72bb0febb2d70796c434d052f04a581d5226e348",
    "generator_script": "0b0d81e9be6ac93b9b7fe33914639454a3350e72489cf4b348a30988f9f72f58",
    "infer_state_effect_prior_helper": "36f7dfca683624e49a7adca9f0c6780e7793705dde77de7f7ba90316be619c87",
    "panel_csv": "7ddbd3b992f7f4630fd4c9b12263c62866a0ab2f7180ccad59a986eea91847d4",
    "panel_manifest": "e6d3ee4956e29277350f3962d60d79aac93cc4f0be267e71603f5218cd7a8960",
    "perturbation_map": "6ab08c62dd117012b386051a9f375505abcd17b88ed76f0ad232199624291170",
    "residual_json": "da5a49a00f4792067f6f12fc95c0b9f5a20a1b52950cb233d2785e0aef2b2636",
    "residual_npz": "dfa24c5443bcfdc99e1879de017a1069b53b7534ea8e605d917e0c765cc25e75",
    "selection_json": "63cc3440c4cab6609ff24fdae0199080732397734e4329aba8c49239fd38d9f1",
    "state_config": "5e0ab29a4af94a5bc9fcc75653ec4267f021bec352e6a2b74678b34c4853521f",
    "support_genes": "e29ff3512b4c2440759bfb2a34577049b2f49df786b3e19c2a91b565de47b38e",
    "var_dims": "1801b3a4d86b202f7ddb7d09559422deb49f824a996a5357cbfeb6f68d8da120",
}
PINNED_STATE_SOURCE = {
    "repository_commit": "9bbfe78a434a55205e4de834e1ea99f85f7a3add",
    "repository_tree": "50a994e0689354ef7103252f3b35b408e6971f86",
}
RUN_EXACT_FIELDS = (
    "axes",
    "checkpoint_selection",
    "data_firewall",
    "model",
    "residual",
)
GROUP_INVARIANT_KEYS = (
    "cells",
    "chunk_sizes",
    "context",
    "control_strata",
    "control_strata_used",
    "count_seed",
    "delta_definition",
    "effect_values",
    "input_library",
    "input_normalization",
    "output_library",
    "paired_passes",
    "panel_ordinal",
    "prediction_ranges",
    "raw_state_delta",
    "residual_nonzero",
    "sample_seed",
    "source_control_index_sha256",
    "target_gene",
    "target_ordinal",
    "target_sum_before",
    "validation_route",
)
PAIR_KEYS = ("g100_vs_g075", "g100_vs_g050", "g075_vs_g050")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "P4 factor contract must end in .json")
    require(not path.exists(), f"Refusing to overwrite P4 factor contract: {path}")
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


def _immutable_descriptor(descriptor: dict[str, Any]) -> dict[str, Any]:
    path, size, digest = _descriptor_identity(descriptor)
    return {"path": path, "size_bytes": size, "sha256": digest}


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized_shared_spec(spec: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(spec)
    normalized.pop("output_tag", None)
    normalized["configuration"].pop(FACTOR_NAME, None)
    normalized["inputs"] = {
        key: (
            None if descriptor is None else _immutable_descriptor(descriptor)
        )
        for key, descriptor in sorted(normalized["inputs"].items())
    }
    return normalized


def _validate_pinned_v51_inputs(spec: dict[str, Any]) -> None:
    inputs = spec.get("inputs", {})
    require(isinstance(inputs, dict), "P4 input descriptors are absent")
    for name, expected_digest in PINNED_V51_INPUT_SHA256.items():
        descriptor = inputs.get(name)
        require(isinstance(descriptor, dict), f"P4 lacks pinned V5.1 input: {name}")
        require(
            str(descriptor.get("sha256", "")) == expected_digest,
            f"P4 input differs from the frozen V5.1 artifact: {name}",
        )
    state_source = spec.get("state_source", {})
    require(isinstance(state_source, dict), "P4 STATE source contract is absent")
    for name, expected_value in PINNED_STATE_SOURCE.items():
        require(
            state_source.get(name) == expected_value,
            f"P4 STATE source differs from the frozen V5.1 runtime: {name}",
        )


def validate_factor_candidates(candidates: dict[str, Candidate]) -> dict[str, Any]:
    """Require a strict, matched three-level STATE-gamma experiment."""

    require(set(candidates) == set(ARM_KEYS), "P4 factor contract requires all three arms")
    reference = candidates["g100"]
    reference_spec = reference.spec
    require(reference_spec.get("schema") == SPEC_SCHEMA, "P4 requires strict-v2 specs")
    require(reference_spec.get("context") == CONTEXT, "P4 is pre-registered for HepG2")
    _validate_pinned_v51_inputs(reference_spec)

    reference_configuration = reference_spec.get("configuration", {})
    reference_inputs = reference_spec.get("inputs", {})
    require(isinstance(reference_configuration, dict), "P4 configuration is absent")
    require(isinstance(reference_inputs, dict), "P4 input descriptors are absent")
    require(
        float(reference_configuration.get("residual_alpha", float("nan"))) == 0.10,
        "P4 requires residual_alpha 0.10",
    )
    require(
        float(
            reference_configuration.get("target_remaining_fraction", float("nan"))
        )
        == 0.20,
        "P4 requires target_remaining_fraction 0.20",
    )
    require(
        float(reference_spec.get("residual_contract", {}).get("shared_response_weight", float("nan")))
        == 0.0,
        "P4 requires shared_response_weight 0.0",
    )

    normalized_reference = _normalized_shared_spec(reference_spec)
    observed_levels: dict[str, float] = {}
    for arm in ARM_KEYS:
        candidate = candidates[arm]
        spec = candidate.spec
        definition = ARM_DEFINITIONS[arm]
        require(spec.get("schema") == SPEC_SCHEMA, f"Arm {arm} is not strict-v2")
        require(spec.get("context") == CONTEXT, f"Arm {arm} context differs")
        require(
            spec.get("output_tag") == definition["output_tag"],
            f"Arm {arm} output tag differs from the pre-registration",
        )
        configuration = spec.get("configuration", {})
        require(
            isinstance(configuration, dict)
            and set(configuration) == set(reference_configuration),
            f"Arm {arm} configuration keys differ",
        )
        observed_weight = float(configuration.get(FACTOR_NAME, float("nan")))
        expected_weight = float(definition[FACTOR_NAME])
        require(
            observed_weight == expected_weight,
            f"Arm {arm} STATE effect weight differs from the pre-registration",
        )
        observed_levels[arm] = observed_weight

        for field in EXACT_SPEC_FIELDS:
            require(
                spec.get(field) == reference_spec.get(field),
                f"Arm {arm} shared specification field differs: {field}",
            )
        inputs = spec.get("inputs", {})
        require(
            isinstance(inputs, dict) and set(inputs) == set(reference_inputs),
            f"Arm {arm} specification input keys differ",
        )
        for name in sorted(reference_inputs):
            left = reference_inputs[name]
            right = inputs[name]
            if left is None or right is None:
                require(
                    left is None and right is None,
                    f"Arm {arm} optional input presence differs: {name}",
                )
            else:
                require(
                    _descriptor_identity(left) == _descriptor_identity(right),
                    f"Arm {arm} generator input differs: {name}",
                )
        require(
            _normalized_shared_spec(spec) == normalized_reference,
            f"Arm {arm} differs outside {FACTOR_NAME}",
        )

    require(
        set(observed_levels.values()) == {0.50, 0.75, 1.00},
        "P4 STATE effect weight levels are incomplete",
    )
    return {
        "context": CONTEXT,
        "factor_name": FACTOR_NAME,
        "levels_by_arm": observed_levels,
        "sole_configuration_difference": FACTOR_NAME,
        "shared_spec_identity_sha256": _canonical_sha256(normalized_reference),
        "pinned_v51_input_sha256": dict(PINNED_V51_INPUT_SHA256),
        "pinned_state_source": dict(PINNED_STATE_SOURCE),
    }


def _generation_identity(candidate: Candidate) -> dict[str, Any]:
    generation = candidate.generation
    identity: dict[str, Any] = {}
    for field in RUN_EXACT_FIELDS:
        require(field in generation, f"Generation report lacks P4 field: {field}")
        identity[field] = generation[field]
    validation = generation.get("validation", {})
    require(isinstance(validation, dict), "Generation validation report is absent")
    identity["validation"] = {
        key: validation.get(key)
        for key in ("checks", "data_dtype", "failed_checks", "groups", "shape")
    }
    groups = generation.get("qc", {}).get("groups")
    require(isinstance(groups, list) and bool(groups), "Generation group QC is absent")
    normalized_groups: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        require(isinstance(group, dict), f"Generation group {index} QC is invalid")
        missing = sorted(set(GROUP_INVARIANT_KEYS) - set(group))
        require(not missing, f"Generation group {index} lacks invariants: {missing}")
        normalized_groups.append({key: group[key] for key in GROUP_INVARIANT_KEYS})
    identity["groups"] = normalized_groups
    return identity


def validate_realized_predictions(
    candidates: dict[str, Candidate], *, chunk_rows: int = 400
) -> dict[str, Any]:
    """Validate realized metadata, inference identity, and count differences."""

    require(chunk_rows > 0, "P4 count-audit chunk size must be positive")
    identities = {arm: _generation_identity(candidates[arm]) for arm in ARM_KEYS}
    reference_identity = identities["g100"]
    for arm in ("g075", "g050"):
        require(
            identities[arm] == reference_identity,
            f"Arm {arm} realized inference or source-control identity differs",
        )

    prediction_digests = {
        arm: str(candidates[arm].descriptors["prediction_h5ad"]["sha256"])
        for arm in ARM_KEYS
    }
    require(
        len(set(prediction_digests.values())) == len(ARM_KEYS),
        "P4 prediction files are not pairwise distinct",
    )

    groups = reference_identity["groups"]
    targets = [str(group["target_gene"]) for group in groups]
    routes = np.asarray([str(group["validation_route"]) for group in groups])
    require(len(targets) == len(set(targets)), "P4 target order contains duplicates")
    require(set(routes) == {"direct", "held_target"}, "P4 route axis is invalid")
    direct = routes == "direct"
    cells_per_group = int(
        candidates["g100"].spec["configuration"]["cells_per_group"]
    )

    datasets: dict[str, ad.AnnData] = {}
    try:
        for arm in ARM_KEYS:
            datasets[arm] = ad.read_h5ad(
                candidates[arm].prediction_path, backed="r"
            )
        pair_definitions = {
            "g100_vs_g075": ("g100", "g075"),
            "g100_vs_g050": ("g100", "g050"),
            "g075_vs_g050": ("g075", "g050"),
        }
        pairwise: dict[str, Any] = {}
        for pair, (left_arm, right_arm) in pair_definitions.items():
            axes = validate_axes_and_pairing(
                datasets[left_arm],
                datasets[right_arm],
                context=CONTEXT,
                targets=targets,
                cells_per_group=cells_per_group,
            )
            counts = stream_compare_counts(
                datasets[left_arm],
                datasets[right_arm],
                direct_mask=direct,
                cells_per_group=cells_per_group,
                chunk_rows=chunk_rows,
            )
            differing_entries = sum(
                int(counts[route]["differing_entries"])
                for route in ("direct", "held_target")
            )
            require(
                differing_entries > 0,
                f"P4 pair {pair} has no realized count difference",
            )
            pairwise[pair] = {
                "axes_and_pairing": axes,
                "count_comparison": counts,
                "differing_entries": differing_entries,
            }
    finally:
        for dataset in datasets.values():
            dataset.file.close()

    return {
        "shared_generation_identity_sha256": _canonical_sha256(reference_identity),
        "prediction_sha256_by_arm": prediction_digests,
        "target_axis_sha256": _canonical_sha256(targets),
        "route_axis_sha256": _canonical_sha256(routes.tolist()),
        "chunk_rows": chunk_rows,
        "pairwise": pairwise,
    }


def build_factor_payload(
    candidates: dict[str, Candidate], realized: dict[str, Any]
) -> dict[str, Any]:
    factor = validate_factor_candidates(candidates)
    pairwise = realized.get("pairwise", {}) if isinstance(realized, dict) else None
    require(
        isinstance(pairwise, dict) and set(pairwise) == set(PAIR_KEYS),
        "P4 realized-prediction receipt is incomplete",
    )
    arms: dict[str, Any] = {}
    for arm in ARM_KEYS:
        candidate = candidates[arm]
        missing = sorted(set(PRIMARY_DESCRIPTOR_KEYS) - set(candidate.descriptors))
        require(not missing, f"Arm {arm} lacks authenticated descriptors: {missing}")
        arms[arm] = {
            **ARM_DEFINITIONS[arm],
            "artifacts": {
                key: _immutable_descriptor(candidate.descriptors[key])
                for key in PRIMARY_DESCRIPTOR_KEYS
            },
        }
    return {
        "schema": SCHEMA,
        "status": "passed",
        "builder_script": _immutable_descriptor(describe_file(Path(__file__))),
        "factor": factor,
        "realized_generation": realized,
        "arms": arms,
        "checks": {
            "all_arms_strict_v2": True,
            "all_generation_receipts_passed": True,
            "all_predictions_authenticated": True,
            "all_generator_inputs_authenticated": True,
            "all_shared_spec_fields_identical": True,
            "realized_inference_and_pairing_identical": True,
            "realized_counts_pairwise_distinct": True,
            "sole_configuration_difference_is_state_effect_weight": True,
            "frozen_v51_inputs_exact": True,
            "pre_registered_levels_exact": True,
            "treated_truth_not_loaded": True,
        },
    }


def _authenticate_arms_from_args(args: argparse.Namespace) -> dict[str, Candidate]:
    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    return {
        arm: authenticate_candidate(
            spec_path=getattr(args, f"{arm}_spec"),
            verification_path=getattr(args, f"{arm}_generation_verification"),
            prediction_path=getattr(args, f"{arm}_prediction"),
            cache=cache,
            label=arm,
        )
        for arm in ARM_KEYS
    }


def validate_factor_receipt(path: Path) -> dict[str, Any]:
    """Re-hash every receipt-bound artifact and reproduce the P4 contract."""

    receipt = _load_json(path, "P4 factor contract")
    require(receipt.get("schema") == SCHEMA, "Bad P4 factor-contract schema")
    require(receipt.get("status") == "passed", "P4 factor contract did not pass")
    checks = receipt.get("checks", {})
    require(
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "P4 factor contract contains a failed check",
    )

    cache: dict[tuple[str, int, str], dict[str, Any]] = {}
    builder = receipt.get("builder_script", {})
    _authenticate_file(Path(str(builder.get("path", ""))), builder, "P4 builder", cache)
    arms = receipt.get("arms", {})
    require(isinstance(arms, dict) and set(arms) == set(ARM_KEYS), "Bad P4 arm set")

    candidates: dict[str, Candidate] = {}
    for arm in ARM_KEYS:
        arm_record = arms[arm]
        require(isinstance(arm_record, dict), f"Arm {arm} receipt is invalid")
        artifacts = arm_record.get("artifacts", {})
        require(
            isinstance(artifacts, dict)
            and set(artifacts) == set(PRIMARY_DESCRIPTOR_KEYS),
            f"Arm {arm} receipt artifacts are incomplete",
        )
        for name in PRIMARY_DESCRIPTOR_KEYS:
            descriptor = artifacts[name]
            _authenticate_file(
                Path(str(descriptor.get("path", ""))),
                descriptor,
                f"arm {arm} {name}",
                cache,
            )
        candidates[arm] = authenticate_candidate(
            spec_path=Path(str(artifacts["spec"]["path"])),
            verification_path=Path(str(artifacts["generation_verified"]["path"])),
            prediction_path=Path(str(artifacts["prediction_h5ad"]["path"])),
            cache=cache,
            label=arm,
        )

    observed = build_factor_payload(candidates, validate_realized_predictions(candidates))
    require(observed == receipt, "P4 factor contract no longer reproduces exactly")
    return receipt


def bind_factor_comparison(
    receipt: dict[str, Any],
    *,
    baseline_label: str,
    baseline_spec: Path | None,
    candidate_label: str,
    candidate_spec: Path | None,
) -> dict[str, Any]:
    """Bind one strict comparison to two named arms in a validated receipt."""

    require(baseline_spec is not None, "Factor-bound baseline requires a strict spec")
    require(candidate_spec is not None, "Factor-bound candidate requires a strict spec")
    by_tag = {
        arm["output_tag"]: arm
        for arm in receipt.get("arms", {}).values()
    }
    require(baseline_label in by_tag, "Baseline label is absent from P4 factor contract")
    require(candidate_label in by_tag, "Candidate label is absent from P4 factor contract")
    require(baseline_label != candidate_label, "P4 factor comparison labels must differ")
    for label, path in (
        (baseline_label, baseline_spec),
        (candidate_label, candidate_spec),
    ):
        expected = by_tag[label]["artifacts"]["spec"]
        require(
            Path(str(expected["path"])).resolve() == path.resolve(),
            f"Comparison spec path differs from P4 factor contract: {label}",
        )
    return {
        "schema": receipt["schema"],
        "factor_name": receipt["factor"]["factor_name"],
        "sole_configuration_difference": receipt["factor"][
            "sole_configuration_difference"
        ],
        "baseline_level": float(by_tag[baseline_label][FACTOR_NAME]),
        "candidate_level": float(by_tag[candidate_label][FACTOR_NAME]),
        "shared_spec_identity_sha256": receipt["factor"][
            "shared_spec_identity_sha256"
        ],
        "shared_generation_identity_sha256": receipt["realized_generation"][
            "shared_generation_identity_sha256"
        ],
        "realized_pairwise_count_audit": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build an immutable P4 factor receipt.")
    for arm in ARM_KEYS:
        build.add_argument(f"--{arm}-spec", type=Path, required=True)
        build.add_argument(
            f"--{arm}-generation-verification", type=Path, required=True
        )
        build.add_argument(f"--{arm}-prediction", type=Path, required=True)
    build.add_argument("--output-json", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="Re-authenticate an existing P4 factor receipt."
    )
    validate.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        candidates = _authenticate_arms_from_args(args)
        payload = build_factor_payload(
            candidates, validate_realized_predictions(candidates)
        )
        _atomic_json(args.output_json, payload)
    else:
        payload = validate_factor_receipt(args.receipt)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": payload["status"],
                "factor": payload["factor"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
