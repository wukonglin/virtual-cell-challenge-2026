#!/usr/bin/env python3
"""Select one P4 STATE-gamma arm from authenticated sequential comparisons.

The selector is deliberately a small, deterministic receipt builder. It
re-authenticates the strict three-arm generation contract and the immutable
comparison JSON files. The factor validator streams generated prediction
counts to reproduce the three-arm audit; neither component reads truth. The
registered decision tree is:

* retain g100 when neither lower-gamma arm passes against g100;
* select the sole passing lower-gamma arm when exactly one passes; and
* when both pass, require g050 versus g075 and select g050 only when that
  incremental experiment gate passes, otherwise select g075.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from build_public_v6_p4_factor_contract import (
    SCHEMA as FACTOR_SCHEMA,
    validate_factor_receipt,
)
from compare_public_v6_sequential import (
    PRIMARY_REQUIRED_GUARDRAILS,
    SCHEMA as COMPARISON_SCHEMA,
    _primary_promotion_gate,
)
from infer_state_effect_prior import require, sha256_file


SCHEMA = "vcc-public-v6-p4-state-gamma-selection-v1"
CONTEXT = "HepG2"
ROLE = "primary"
FACTOR_NAME = "state_effect_weight"
EXPECTED_COHORT_SIZES = {"all": 300, "direct": 267, "held_target": 33}
ARMS = {
    "g100": {"label": "p4_g100_p0_a010", "state_effect_weight": 1.00},
    "g075": {"label": "p4_g075_p0_a010", "state_effect_weight": 0.75},
    "g050": {"label": "p4_g050_p0_a010", "state_effect_weight": 0.50},
}
COMPARISONS = {
    "g075_vs_g100": {"baseline_arm": "g100", "candidate_arm": "g075"},
    "g050_vs_g100": {"baseline_arm": "g100", "candidate_arm": "g050"},
    "g050_vs_g075": {"baseline_arm": "g075", "candidate_arm": "g050"},
}
PRIMARY_COMPARISONS = ("g075_vs_g100", "g050_vs_g100")
INCREMENTAL_COMPARISON = "g050_vs_g075"
DECISION_RULE = (
    "retain_g100_if_neither_primary_gate_passes;select_the_only_passing_"
    "lower_gamma_arm_if_exactly_one_passes;if_both_pass_require_g050_vs_"
    "g075_and_select_g050_only_if_its_incremental_experiment_gate_passes_"
    "else_select_g075"
)


def _load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"Missing {label}: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Invalid {label}: {path}") from error
    require(isinstance(payload, dict), f"{label} is not a JSON object")
    return payload


def _immutable_descriptor(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"Missing receipt-bound file: {resolved}")
    return {
        "path": str(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "sha256": sha256_file(resolved),
    }


def _descriptor_identity(descriptor: Any, label: str) -> tuple[str, int, str]:
    require(isinstance(descriptor, dict), f"{label} descriptor is not an object")
    path = Path(str(descriptor.get("path", ""))).resolve()
    try:
        size = int(descriptor.get("size_bytes", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} descriptor size is invalid") from error
    digest = str(descriptor.get("sha256", ""))
    require(size >= 0, f"{label} descriptor size is invalid")
    require(
        len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label} descriptor SHA-256 is invalid",
    )
    return str(path), size, digest


def _authenticate_descriptor(descriptor: Any, label: str) -> Path:
    path_string, expected_size, expected_digest = _descriptor_identity(
        descriptor, label
    )
    path = Path(path_string)
    require(path.is_file(), f"Missing {label}: {path}")
    require(path.stat().st_size == expected_size, f"{label} size differs")
    require(sha256_file(path) == expected_digest, f"{label} SHA-256 differs")
    return path


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "P4 selection receipt must end in .json")
    require(not path.exists(), f"Refusing to overwrite P4 selection receipt: {path}")
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


def _validate_factor_shape(receipt: dict[str, Any]) -> dict[str, Any]:
    """Fail closed if the authenticated factor receipt is not the P4 grid."""

    require(receipt.get("schema") == FACTOR_SCHEMA, "Bad P4 factor schema")
    require(receipt.get("status") == "passed", "P4 factor receipt did not pass")
    factor = receipt.get("factor")
    require(isinstance(factor, dict), "P4 factor definition is absent")
    require(factor.get("context") == CONTEXT, "P4 factor context is not HepG2")
    require(factor.get("factor_name") == FACTOR_NAME, "P4 factor name differs")
    require(
        factor.get("sole_configuration_difference") == FACTOR_NAME,
        "P4 arms differ outside state_effect_weight",
    )
    expected_levels = {
        arm: definition[FACTOR_NAME] for arm, definition in ARMS.items()
    }
    require(
        factor.get("levels_by_arm") == expected_levels,
        "P4 factor levels differ from the pre-registration",
    )
    shared_spec = str(factor.get("shared_spec_identity_sha256", ""))
    require(len(shared_spec) == 64, "P4 shared specification identity is invalid")

    receipt_arms = receipt.get("arms")
    require(
        isinstance(receipt_arms, dict) and set(receipt_arms) == set(ARMS),
        "P4 factor arm set differs",
    )
    for arm, expected in ARMS.items():
        observed = receipt_arms[arm]
        require(isinstance(observed, dict), f"P4 factor arm {arm} is invalid")
        require(
            observed.get("output_tag") == expected["label"],
            f"P4 factor arm {arm} label differs",
        )
        require(
            observed.get(FACTOR_NAME) == expected[FACTOR_NAME],
            f"P4 factor arm {arm} level differs",
        )

    realized = receipt.get("realized_generation")
    require(isinstance(realized, dict), "P4 realized-generation receipt is absent")
    shared_generation = str(
        realized.get("shared_generation_identity_sha256", "")
    )
    require(
        len(shared_generation) == 64,
        "P4 shared generation identity is invalid",
    )
    return {
        "context": CONTEXT,
        "factor_name": FACTOR_NAME,
        "levels_by_arm": expected_levels,
        "levels_by_label": {
            definition["label"]: definition[FACTOR_NAME]
            for definition in ARMS.values()
        },
        "sole_configuration_difference": FACTOR_NAME,
        "shared_spec_identity_sha256": shared_spec,
        "shared_generation_identity_sha256": shared_generation,
    }


def _expected_factor_binding(
    factor_receipt: dict[str, Any],
    factor_descriptor: dict[str, Any],
    baseline_arm: str,
    candidate_arm: str,
) -> dict[str, Any]:
    factor = factor_receipt["factor"]
    realized = factor_receipt["realized_generation"]
    return {
        "receipt": factor_descriptor,
        "schema": FACTOR_SCHEMA,
        "factor_name": FACTOR_NAME,
        "sole_configuration_difference": FACTOR_NAME,
        "baseline_level": ARMS[baseline_arm][FACTOR_NAME],
        "candidate_level": ARMS[candidate_arm][FACTOR_NAME],
        "shared_spec_identity_sha256": factor["shared_spec_identity_sha256"],
        "shared_generation_identity_sha256": realized[
            "shared_generation_identity_sha256"
        ],
        "realized_pairwise_count_audit": True,
    }


def _shared_evaluation_identity(report: dict[str, Any]) -> dict[str, Any]:
    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "Comparison provenance is absent")
    manifest = provenance.get("manifest")
    manifest_identity = _descriptor_identity(manifest, "comparison manifest")
    _authenticate_descriptor(manifest, "comparison manifest")
    shared_scoring = provenance.get("shared_scoring_identity")
    require(
        isinstance(shared_scoring, dict) and bool(shared_scoring),
        "Comparison shared scoring identity is absent",
    )
    configuration = report.get("configuration")
    require(
        isinstance(configuration, dict) and bool(configuration),
        "Comparison configuration is absent",
    )
    matched_nmae = report.get("matched_nmae_subset")
    require(
        isinstance(matched_nmae, dict) and matched_nmae.get("matched") is True,
        "Comparison NMAE population is not matched",
    )
    uncertainty = report.get("uncertainty_contract")
    require(
        isinstance(uncertainty, dict)
        and bool(uncertainty)
        and all(value is True for value in uncertainty.values()),
        "Comparison uncertainty contract is incomplete",
    )
    return {
        "manifest": {
            "path": manifest_identity[0],
            "size_bytes": manifest_identity[1],
            "sha256": manifest_identity[2],
        },
        "shared_scoring_identity": shared_scoring,
        "configuration": configuration,
        "cohort_sizes": report["cohort_sizes"],
        "matched_nmae_subset": matched_nmae,
        "uncertainty_contract": uncertainty,
    }


def _arm_scoring_identities(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Bind each arm to the exact scored artifacts reused across comparisons."""

    provenance = report.get("provenance")
    require(isinstance(provenance, dict), "Comparison provenance is absent")
    bundles = provenance.get("authenticated_scoring_bundles")
    require(
        isinstance(bundles, dict) and set(bundles) == {"baseline", "candidate"},
        "Comparison scoring bundles are incomplete",
    )
    identities: dict[str, dict[str, Any]] = {}
    for role in ("baseline", "candidate"):
        label = str(report.get(f"{role}_label", ""))
        require(label in {arm["label"] for arm in ARMS.values()}, "Unknown P4 arm label")
        bundle = bundles[role]
        require(isinstance(bundle, dict), f"Comparison {role} scoring bundle is invalid")
        require(
            bundle.get("context") == CONTEXT and bundle.get("label") == label,
            f"Comparison {role} scoring bundle identity differs",
        )
        receipt = bundle.get("receipt")
        require(isinstance(receipt, dict), f"Comparison {role} scoring receipt is absent")
        _authenticate_descriptor(receipt, f"comparison {role} scoring receipt")

        results = provenance.get(f"{role}_results")
        results_path = _authenticate_descriptor(
            results, f"comparison {role} results"
        )
        results_identity = _descriptor_identity(
            results, f"comparison {role} results"
        )
        sidecars = provenance.get(f"{role}_cell_eval2_sidecars")
        require(
            isinstance(sidecars, dict)
            and set(sidecars)
            == {
                "aggregate_results",
                "metric_aggregation",
                "validated_all_context_means",
                "expected_targets",
            },
            f"Comparison {role} cell-eval sidecars are absent",
        )
        authenticated_sidecars = {
            "aggregate_results": _immutable_descriptor(
                _authenticate_descriptor(
                    sidecars["aggregate_results"],
                    f"comparison {role} aggregate results",
                )
            ),
            "metric_aggregation": _immutable_descriptor(
                _authenticate_descriptor(
                    sidecars["metric_aggregation"],
                    f"comparison {role} metric aggregation",
                )
            ),
            "validated_all_context_means": sidecars[
                "validated_all_context_means"
            ],
            "expected_targets": sidecars["expected_targets"],
        }
        identities[label] = {
            "results": {
                "path": str(results_path),
                "size_bytes": results_identity[1],
                "sha256": results_identity[2],
            },
            "authenticated_scoring_bundle": bundle,
            "cell_eval2_sidecars": authenticated_sidecars,
        }
    require(len(identities) == 2, "Comparison does not bind two distinct P4 arms")
    return identities


def _authenticate_comparison(
    *,
    key: str,
    path: Path,
    factor_receipt: dict[str, Any],
    factor_descriptor: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, dict[str, Any]]]:
    """Authenticate one comparison and independently reproduce its hard gate."""

    definition = COMPARISONS[key]
    baseline_arm = definition["baseline_arm"]
    candidate_arm = definition["candidate_arm"]
    report = _load_json(path, f"P4 comparison {key}")
    require(
        report.get("schema") == COMPARISON_SCHEMA,
        f"P4 comparison {key} has the wrong schema",
    )
    require(report.get("context") == CONTEXT, f"P4 comparison {key} is not HepG2")
    require(report.get("role") == ROLE, f"P4 comparison {key} is not primary")
    require(
        report.get("baseline_label") == ARMS[baseline_arm]["label"],
        f"P4 comparison {key} baseline label differs",
    )
    require(
        report.get("candidate_label") == ARMS[candidate_arm]["label"],
        f"P4 comparison {key} candidate label differs",
    )
    require(
        report.get("cohort_sizes") == EXPECTED_COHORT_SIZES,
        f"P4 comparison {key} cohort sizes differ",
    )

    cohorts = report.get("cohorts")
    require(
        isinstance(cohorts, dict)
        and set(cohorts) == {"all", "direct", "held_target"},
        f"P4 comparison {key} cohorts are incomplete",
    )
    for cohort in ("all", "direct"):
        metrics = cohorts[cohort].get("metrics")
        require(
            isinstance(metrics, dict)
            and set(PRIMARY_REQUIRED_GUARDRAILS).issubset(metrics),
            f"P4 comparison {key} {cohort} guardrail metrics are incomplete",
        )
    reproduced_gate = _primary_promotion_gate(cohorts)
    require(
        report.get("primary_promotion_gate") == reproduced_gate,
        f"P4 comparison {key} primary gate does not reproduce",
    )
    expected_experiment_gate = {"role": ROLE, **reproduced_gate}
    require(
        report.get("experiment_gate") == expected_experiment_gate,
        f"P4 comparison {key} experiment gate does not reproduce",
    )
    require(
        isinstance(reproduced_gate.get("passed"), bool),
        f"P4 comparison {key} experiment gate is not Boolean",
    )

    provenance = report.get("provenance")
    require(isinstance(provenance, dict), f"P4 comparison {key} provenance is absent")
    expected_binding = _expected_factor_binding(
        factor_receipt,
        factor_descriptor,
        baseline_arm,
        candidate_arm,
    )
    require(
        provenance.get("factor_contract") == expected_binding,
        f"P4 comparison {key} factor-contract binding differs",
    )

    firewall = report.get("leakage_firewall")
    require(isinstance(firewall, dict), f"P4 comparison {key} firewall is absent")
    require(
        firewall.get("reads_truth_h5ad") is False
        and firewall.get("loads_scoring_view_expression_matrices") is False
        and firewall.get(
            "streams_generated_prediction_counts_for_factor_authentication"
        )
        is True
        and firewall.get("materializes_full_prediction_expression_matrices") is False
        and firewall.get("writes_or_modifies_input_artifacts") is False
        and firewall.get("authenticates_factor_contract") is True,
        f"P4 comparison {key} leakage firewall failed",
    )

    shared_identity = _shared_evaluation_identity(report)
    summary = {
        "file": _immutable_descriptor(path),
        "baseline": {
            "arm": baseline_arm,
            "label": ARMS[baseline_arm]["label"],
            FACTOR_NAME: ARMS[baseline_arm][FACTOR_NAME],
        },
        "candidate": {
            "arm": candidate_arm,
            "label": ARMS[candidate_arm]["label"],
            FACTOR_NAME: ARMS[candidate_arm][FACTOR_NAME],
        },
        "experiment_gate_passed": bool(reproduced_gate["passed"]),
        "shared_evaluation_identity_sha256": _canonical_sha256(shared_identity),
    }
    return summary, shared_identity, _arm_scoring_identities(report)


def build_selection_payload(
    *,
    factor_contract: Path,
    g075_vs_g100: Path,
    g050_vs_g100: Path,
    g050_vs_g075: Path | None = None,
) -> dict[str, Any]:
    """Authenticate P4 evidence and apply the registered decision tree."""

    factor_receipt = validate_factor_receipt(factor_contract)
    factor = _validate_factor_shape(factor_receipt)
    factor_descriptor = _immutable_descriptor(factor_contract)

    comparison_paths = {
        "g075_vs_g100": g075_vs_g100,
        "g050_vs_g100": g050_vs_g100,
    }
    comparisons: dict[str, Any] = {}
    shared_identities: dict[str, dict[str, Any]] = {}
    arm_scoring_identities: dict[str, dict[str, Any]] = {}

    def record_comparison(key: str, path: Path) -> None:
        summary, shared_identity, observed_arms = _authenticate_comparison(
            key=key,
            path=path,
            factor_receipt=factor_receipt,
            factor_descriptor=factor_descriptor,
        )
        comparisons[key] = summary
        shared_identities[key] = shared_identity
        for label, identity in observed_arms.items():
            if label in arm_scoring_identities:
                require(
                    arm_scoring_identities[label] == identity,
                    f"P4 arm {label} uses different scored artifacts across comparisons",
                )
            else:
                arm_scoring_identities[label] = identity

    for key in PRIMARY_COMPARISONS:
        record_comparison(key, comparison_paths[key])

    primary_passes = {
        "g075": comparisons["g075_vs_g100"]["experiment_gate_passed"],
        "g050": comparisons["g050_vs_g100"]["experiment_gate_passed"],
    }
    incremental_required = bool(primary_passes["g075"] and primary_passes["g050"])
    if incremental_required:
        require(
            g050_vs_g075 is not None,
            "Both lower-gamma arms passed; g050-vs-g075 is required",
        )
        record_comparison(INCREMENTAL_COMPARISON, g050_vs_g075)
        incremental_passed: bool | None = comparisons[INCREMENTAL_COMPARISON][
            "experiment_gate_passed"
        ]
        selected_arm = "g050" if incremental_passed else "g075"
    else:
        require(
            g050_vs_g075 is None,
            "g050-vs-g075 is only admissible when both primary arms pass",
        )
        comparisons[INCREMENTAL_COMPARISON] = None
        incremental_passed = None
        if primary_passes["g075"]:
            selected_arm = "g075"
        elif primary_passes["g050"]:
            selected_arm = "g050"
        else:
            selected_arm = "g100"

    identity_digests = {
        key: _canonical_sha256(value) for key, value in shared_identities.items()
    }
    require(
        len(set(identity_digests.values())) == 1,
        "P4 comparisons do not share one evaluation identity",
    )
    common_identity = next(iter(shared_identities.values()))
    selected = ARMS[selected_arm]
    return {
        "schema": SCHEMA,
        "status": "passed",
        "builder_script": _immutable_descriptor(Path(__file__)),
        "factor_contract": factor_descriptor,
        "factor": factor,
        "comparisons": comparisons,
        "shared_evaluation_identity": common_identity,
        "arm_scoring_identities": arm_scoring_identities,
        "decision": {
            "rule": DECISION_RULE,
            "primary_experiment_gate_passed": primary_passes,
            "incremental_comparison_required": incremental_required,
            "incremental_experiment_gate_passed": incremental_passed,
            "selected_arm": selected_arm,
            "selected_label": selected["label"],
            FACTOR_NAME: selected[FACTOR_NAME],
        },
        "checks": {
            "factor_receipt_reauthenticated": True,
            "factor_levels_and_labels_exact": True,
            "primary_comparisons_authenticated": True,
            "comparison_gates_reproduced_from_cohorts": True,
            "factor_contract_descriptor_and_digest_bound": True,
            "all_comparisons_share_evaluation_identity": True,
            "repeated_arms_reuse_exact_scored_artifacts": True,
            "manifest_results_receipts_and_sidecars_rehashed": True,
            "incremental_comparison_authenticated_if_required": True,
            "registered_decision_tree_applied": True,
            "truth_matrices_not_read": True,
            "prediction_matrices_not_read_outside_factor_validator": True,
        },
        "data_firewall": {
            "reads_factor_contract_json": True,
            "factor_validator_may_stream_prediction_matrices": True,
            "reads_sequential_comparison_json": True,
            "rehashes_manifest_results_scoring_receipts_and_sidecars": True,
            "reads_scoring_view_h5ad_outside_factor_validator": False,
            "reads_truth_h5ad": False,
            "loads_truth_expression_matrices": False,
            "reads_prediction_h5ad_outside_factor_validator": False,
            "loads_prediction_expression_matrices_outside_factor_validator": False,
            "writes_or_modifies_input_artifacts": False,
        },
    }


def validate_selection_receipt(path: Path) -> dict[str, Any]:
    """Re-hash all inputs and exactly reproduce one selection receipt."""

    receipt = _load_json(path, "P4 selection receipt")
    require(receipt.get("schema") == SCHEMA, "Bad P4 selection receipt schema")
    require(receipt.get("status") == "passed", "P4 selection receipt did not pass")
    checks = receipt.get("checks")
    require(
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        "P4 selection receipt contains a failed check",
    )

    builder_path = _authenticate_descriptor(
        receipt.get("builder_script"), "P4 selection builder"
    )
    require(
        builder_path == Path(__file__).resolve(),
        "P4 selection receipt was built by a different script path",
    )
    factor_path = _authenticate_descriptor(
        receipt.get("factor_contract"), "P4 factor contract"
    )
    comparisons = receipt.get("comparisons")
    require(
        isinstance(comparisons, dict) and set(comparisons) == set(COMPARISONS),
        "P4 selection comparison set is invalid",
    )
    paths: dict[str, Path | None] = {}
    for key in PRIMARY_COMPARISONS:
        record = comparisons[key]
        require(isinstance(record, dict), f"P4 selection lacks {key}")
        paths[key] = _authenticate_descriptor(
            record.get("file"), f"P4 comparison {key}"
        )
    incremental = comparisons[INCREMENTAL_COMPARISON]
    if incremental is None:
        paths[INCREMENTAL_COMPARISON] = None
    else:
        require(
            isinstance(incremental, dict),
            "P4 incremental comparison record is invalid",
        )
        paths[INCREMENTAL_COMPARISON] = _authenticate_descriptor(
            incremental.get("file"), f"P4 comparison {INCREMENTAL_COMPARISON}"
        )

    reproduced = build_selection_payload(
        factor_contract=factor_path,
        g075_vs_g100=paths["g075_vs_g100"],  # type: ignore[arg-type]
        g050_vs_g100=paths["g050_vs_g100"],  # type: ignore[arg-type]
        g050_vs_g075=paths["g050_vs_g075"],
    )
    require(
        reproduced == receipt,
        "P4 selection receipt no longer reproduces exactly",
    )
    return receipt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build", help="Build an immutable P4 selection receipt.")
    build.add_argument("--factor-contract", type=Path, required=True)
    build.add_argument("--g075-vs-g100", type=Path, required=True)
    build.add_argument("--g050-vs-g100", type=Path, required=True)
    build.add_argument("--g050-vs-g075", type=Path)
    build.add_argument("--output-json", type=Path, required=True)

    validate = commands.add_parser(
        "validate", help="Re-authenticate and reproduce a P4 selection receipt."
    )
    validate.add_argument("--receipt", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        payload = build_selection_payload(
            factor_contract=args.factor_contract,
            g075_vs_g100=args.g075_vs_g100,
            g050_vs_g100=args.g050_vs_g100,
            g050_vs_g075=args.g050_vs_g075,
        )
        _atomic_json(args.output_json, payload)
    else:
        payload = validate_selection_receipt(args.receipt)
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "status": payload["status"],
                "decision": payload["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
