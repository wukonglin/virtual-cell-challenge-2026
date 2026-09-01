#!/usr/bin/env python3
"""Audit whether the pinned upstream scDFM artifact can satisfy VCC inputs.

This audit intentionally reads only small manifests, CSV axes, vocabulary JSON,
the authenticated P4 selection receipt, and the pinned source checkout. It does
not load challenge controls, public treated matrices, or prediction matrices.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tomllib
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from authenticate_scdfm_source import (
    DEFAULT_SOURCE,
    authenticate_source,
    write_json_atomic,
)


SCHEMA = "vcc-scdfm-compatibility-audit-v2"
EXPECTED_CONFIG_SCHEMA = "vcc-public-v7-scdfm-experiment-v1"
DEFAULT_TARGETS = Path("dataset/controls/pert_counts.csv")
DEFAULT_GENE_AXIS = Path("dataset/controls/gene_names.csv")
DEFAULT_VOCAB = DEFAULT_SOURCE / "src/tokenizer/norman_5000_highly_vocab.json"
DEFAULT_P4_SELECTION = Path(
    "artifacts/public_v6/decisions/hepg2_p4_state_gamma.json"
)
DEFAULT_CONFIG = Path("configs/scdfm/vcc2026_v7_gamma1.toml")
EXPECTED_P4_SHA256 = (
    "cd6618f38f0b93c8e7b3c63a8a2c1a31b87360ead5cc91b3f7e43898c100d312"
)
SPECIAL_TOKENS = frozenset({"<pad>", "<cls>", "<mask>", "control"})


class CompatibilityError(RuntimeError):
    """Raised when an input violates the registered compatibility contract."""


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    """Return the SHA-256 digest of a regular file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CompatibilityError(message)


def read_unique_csv_column(path: Path, column: str) -> list[str]:
    """Read a required non-empty CSV column and reject duplicate identifiers."""

    _require(path.is_file(), f"Missing CSV input: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None, f"Missing CSV header: {path}")
        _require(column in reader.fieldnames, f"Missing column {column!r}: {path}")
        values = []
        for row_number, row in enumerate(reader, start=2):
            value = (row.get(column) or "").strip()
            _require(value, f"Empty {column!r} at {path}:{row_number}")
            values.append(value)
    _require(values, f"No values in {column!r}: {path}")
    duplicate_count = len(values) - len(set(values))
    _require(
        duplicate_count == 0,
        f"Duplicate {column!r} values in {path}: {duplicate_count}",
    )
    return values


def read_vocab(path: Path) -> dict[str, int]:
    """Read and validate the upstream token-to-index vocabulary."""

    _require(path.is_file(), f"Missing scDFM vocabulary: {path}")
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(isinstance(payload, dict) and payload, "Vocabulary must be an object")
    vocab: dict[str, int] = {}
    for token, index in payload.items():
        _require(isinstance(token, str) and token, "Vocabulary token must be non-empty")
        _require(
            isinstance(index, int) and not isinstance(index, bool) and index >= 0,
            f"Invalid vocabulary index for {token!r}: {index!r}",
        )
        vocab[token] = index
    _require(len(set(vocab.values())) == len(vocab), "Vocabulary indices are not unique")
    _require(
        set(vocab.values()) == set(range(len(vocab))),
        "Vocabulary indices must be consecutive from zero",
    )
    return vocab


def authenticate_p4_selection(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Authenticate the P4 decision and require the immutable gamma-one arm."""

    _require(path.is_file(), f"Missing P4 selection receipt: {path}")
    digest = sha256_file(path)
    _require(
        digest == expected_sha256,
        f"Unexpected P4 selection SHA-256: {digest}; expected {expected_sha256}",
    )
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    _require(payload.get("status") == "passed", "P4 selection did not pass")
    decision = payload.get("decision")
    _require(isinstance(decision, dict), "P4 selection is missing its decision")
    _require(decision.get("selected_arm") == "g100", "P4 did not select g100")
    weight = decision.get("state_effect_weight")
    _require(
        isinstance(weight, (int, float))
        and not isinstance(weight, bool)
        and float(weight) == 1.0,
        "P4 state_effect_weight must be exactly 1.0",
    )
    return {
        "path": str(path.resolve()),
        "sha256": digest,
        "selected_arm": "g100",
        "state_effect_weight": 1.0,
    }


def authenticate_adapter_contract(
    path: Path,
    *,
    p4_selection_path: Path,
    expected_p4_sha256: str,
) -> dict[str, Any]:
    """Authenticate the fail-closed V7 adapter fields from its TOML."""

    _require(path.is_file(), f"Missing V7 adapter config: {path}")
    try:
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        experiment = payload["experiment"]
        anchor = payload["anchor"]
        challenge = payload["challenge"]
        model = payload["model"]
    except (OSError, KeyError, TypeError, tomllib.TOMLDecodeError) as error:
        raise CompatibilityError(f"Invalid V7 adapter config: {error}") from error

    _require(
        experiment.get("schema") == EXPECTED_CONFIG_SCHEMA,
        "V7 config schema is missing or unsupported",
    )
    _require(
        experiment.get("official_submission_allowed") is False,
        "V7 config must forbid official submission",
    )
    _require(anchor.get("selected_arm") == "g100", "V7 anchor must select g100")
    weight = anchor.get("state_effect_weight")
    _require(
        isinstance(weight, (int, float))
        and not isinstance(weight, bool)
        and float(weight) == 1.0,
        "V7 config state_effect_weight must be exactly 1.0",
    )
    _require(
        anchor.get("immutable_during_scdfm_search") is True,
        "V7 STATE anchor must be immutable",
    )
    registered_selection = anchor.get("selection_receipt")
    _require(
        isinstance(registered_selection, str)
        and Path(registered_selection).resolve() == p4_selection_path.resolve(),
        "V7 config selection receipt does not match the audited P4 receipt",
    )
    _require(
        anchor.get("selection_receipt_sha256") == expected_p4_sha256,
        "V7 config P4 receipt SHA-256 does not match the audited digest",
    )
    _require(
        model.get("role") == "reliability_gated_residual_distribution",
        "V7 config must register scDFM as a residual distribution model",
    )
    _require(
        model.get("absolute_scdfm_output_allowed") is False,
        "V7 config must forbid absolute scDFM output",
    )
    _require(
        model.get("center_residual_by_context_target") is True,
        "V7 config must require context-target residual centering",
    )
    _require(
        challenge.get("measured_challenge_perturbation_data_allowed") is False,
        "V7 config must forbid measured challenge perturbation data",
    )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "config_schema": EXPECTED_CONFIG_SCHEMA,
        "config_sha256_bound": True,
        "official_submission_allowed": False,
        "state_effect_weight": 1.0,
        "state_anchor_immutable": True,
        "scdfm_role": model["role"],
        "absolute_scdfm_output_allowed": False,
        "center_residual_by_context_target": True,
        "measured_challenge_perturbation_data_allowed": False,
    }


def _coverage(reference: Sequence[str], available: set[str]) -> dict[str, Any]:
    covered = sorted(set(reference) & available)
    missing = sorted(set(reference) - available)
    return {
        "reference_count": len(reference),
        "covered_count": len(covered),
        "missing_count": len(missing),
        "coverage_fraction": len(covered) / len(reference),
        "covered": covered,
        "missing": missing,
    }


def audit_compatibility(
    *,
    source: Path,
    targets_path: Path,
    gene_axis_path: Path,
    vocab_path: Path,
    p4_selection_path: Path,
    expected_p4_sha256: str,
    config_path: Path,
    expected_target_count: int = 300,
    expected_gene_count: int = 18533,
    source_authenticator: Callable[[Path], Mapping[str, Any]] = authenticate_source,
) -> dict[str, Any]:
    """Build a deterministic compatibility receipt from authenticated inputs."""

    source_receipt = dict(source_authenticator(source))
    _require(source_receipt.get("status") == "passed", "scDFM source audit failed")
    targets = read_unique_csv_column(targets_path, "target_gene")
    gene_axis = read_unique_csv_column(gene_axis_path, "gene_name")
    _require(
        len(targets) == expected_target_count,
        f"Expected {expected_target_count} targets, found {len(targets)}",
    )
    _require(
        len(gene_axis) == expected_gene_count,
        f"Expected {expected_gene_count} genes, found {len(gene_axis)}",
    )
    vocab = read_vocab(vocab_path)
    p4 = authenticate_p4_selection(p4_selection_path, expected_p4_sha256)
    adapter_contract = authenticate_adapter_contract(
        config_path,
        p4_selection_path=p4_selection_path,
        expected_p4_sha256=expected_p4_sha256,
    )

    gene_tokens = set(vocab) - SPECIAL_TOKENS
    target_coverage = _coverage(targets, gene_tokens)
    gene_coverage = _coverage(gene_axis, gene_tokens)
    eligibility_checks = {
        "all_vcc_targets_have_distinct_native_tokens": (
            target_coverage["missing_count"] == 0
        ),
        "full_vcc_gene_axis_is_native": gene_coverage["missing_count"] == 0,
        "native_output_is_raw_integer_counts": False,
        "native_output_has_exactly_400_cells_per_group": False,
        "published_validation_holds_out_complete_cell_lines": False,
        "published_genetic_modality_matches_crispri": False,
    }
    eligible = all(eligibility_checks.values())

    return {
        "schema": SCHEMA,
        "status": "passed",
        "receipt_contract": {
            "adapter_config_sha256_required": True,
            "legacy_unbound_adapter_receipts_accepted": False,
        },
        "anchor": p4,
        "source_authentication": source_receipt,
        "inputs": {
            "targets": {
                "path": str(targets_path.resolve()),
                "sha256": sha256_file(targets_path),
            },
            "gene_axis": {
                "path": str(gene_axis_path.resolve()),
                "sha256": sha256_file(gene_axis_path),
            },
            "norman_vocab": {
                "path": str(vocab_path.resolve()),
                "sha256": sha256_file(vocab_path),
                "token_count": len(vocab),
                "gene_token_count": len(gene_tokens),
                "special_tokens": sorted(set(vocab) & SPECIAL_TOKENS),
            },
            "adapter_config": {
                "path": adapter_contract["path"],
                "sha256": adapter_contract["sha256"],
            },
        },
        "coverage": {
            "vcc_targets_in_norman_vocab": target_coverage,
            "vcc_gene_axis_in_norman_vocab": gene_coverage,
        },
        "direct_checkpoint_eligibility": {
            "eligible": eligible,
            "checks": eligibility_checks,
            "decision": (
                "eligible_for_direct_vcc_inference"
                if eligible
                else "not_eligible_requires_vcc_adapter_and_retraining"
            ),
        },
        "authenticated_adapter_boundary": adapter_contract,
        "upstream_interface_facts": {
            "genetic_training_context": "K562",
            "genetic_training_modality": "CRISPRa",
            "published_evaluation_gene_subset": 1000,
            "native_output_space": "continuous_log_normalized_gene_subset",
            "paper_demonstrates_cross_cell_line_zero_shot": False,
            "paper_demonstrates_crispri_transfer": False,
        },
        "data_firewall": {
            "reads_challenge_control_expression": False,
            "reads_challenge_treated_expression": False,
            "reads_public_treated_expression": False,
            "reads_prediction_expression": False,
            "reads_only_axes_vocab_source_selection_and_config": True,
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit pinned scDFM compatibility with the VCC 2026 contract."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--gene-axis", type=Path, default=DEFAULT_GENE_AXIS)
    parser.add_argument("--vocab", type=Path, default=DEFAULT_VOCAB)
    parser.add_argument("--p4-selection", type=Path, default=DEFAULT_P4_SELECTION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--expected-p4-sha256", default=EXPECTED_P4_SHA256, help=argparse.SUPPRESS
    )
    parser.add_argument("--expected-target-count", type=int, default=300)
    parser.add_argument("--expected-gene-count", type=int, default=18533)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = audit_compatibility(
        source=args.source,
        targets_path=args.targets,
        gene_axis_path=args.gene_axis,
        vocab_path=args.vocab,
        p4_selection_path=args.p4_selection,
        expected_p4_sha256=args.expected_p4_sha256,
        config_path=args.config,
        expected_target_count=args.expected_target_count,
        expected_gene_count=args.expected_gene_count,
    )
    if args.output is not None:
        write_json_atomic(args.output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
