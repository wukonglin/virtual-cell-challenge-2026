#!/usr/bin/env python3
"""Authenticate one immutable public-validation scorer bundle.

The V5.1 scorer stores both reference arms in one completed context receipt,
whereas V6 stores one verification receipt per generated candidate.  This
module normalizes those two receipt formats into one identity record.  It
re-hashes the consumed result table and every adjacent scorer sidecar, checks
the recorded run configuration, and binds V6 results to the exact candidate
specification that produced them.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from public_candidate_v6_contract import (
    LEGACY_SPEC_SCHEMA as LEGACY_V6_SPEC_SCHEMA,
    SPEC_SCHEMA as V6_SPEC_SCHEMA,
    _validate_strict_spec_provenance,
)


V51_RECEIPT_SCHEMA = "vcc-public-validation-scoring-receipt-v1"
V6_RECEIPT_SCHEMA = "vcc-public-cell-eval-run-verification-v1"
V6_GENERATION_VERIFICATION_SCHEMA = "vcc-public-candidate-verification-v1"
EXPECTED_CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
EXPECTED_CELL_EVAL_VERSION = "0.16.0"
EXPECTED_PDEX_VERSION = "0.3.0"
V51_ROLES = ("anchor", "v51_alpha010")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")

DESCRIPTOR_BASE_FIELDS = {"path", "size_bytes", "sha256"}
DESCRIPTOR_WITH_MTIME_FIELDS = DESCRIPTOR_BASE_FIELDS | {"mtime_ns"}
V6_TOP_LEVEL_FIELDS = {
    "candidate_receipts",
    "context",
    "contract",
    "metrics",
    "output_tag",
    "outputs",
    "resolved_run",
    "schema",
    "software",
    "status",
}
V6_OUTPUT_FIELDS = {
    "aggregate_results",
    "expected_targets",
    "metric_aggregation",
    "results",
    "run_meta",
    "validated_all_context_means",
}
V6_CANDIDATE_RECEIPT_BASE_FIELDS = {"spec", "generation_verification", "views"}
V6_CANDIDATE_RECEIPT_STRICT_FIELDS = V6_CANDIDATE_RECEIPT_BASE_FIELDS | {
    "prediction_view",
    "truth_view",
}
VIEWS_FIELDS = {"schema", "context", "axes", "contract", "provenance"}
VIEWS_PROVENANCE_FIELDS = {
    "common_genes",
    "common_genes_json",
    "controls_only",
    "output_prediction",
    "output_truth",
    "prediction",
    "sealed_truth",
}
V6_CONTRACT_FIELDS = {
    "aggregate_mse_matches_cell_eval2",
    "derived_mse_reconstructed_from_complete_components",
    "direction_nan_targets_are_retained_and_counted",
    "metric_aggregation_n_used_authenticated",
    "nmae_real_side_omission_is_explicit",
}
V51_TOP_LEVEL_FIELDS = {
    "schema",
    "status",
    "context",
    "full_generation_summary",
    "predictions",
    "generation_qc",
    "scoring_software",
    "resolved_scoring",
    "outputs",
}
V51_OUTPUT_FIELDS = {
    "views_report",
    "results",
    "aggregate_results",
    "metric_aggregation",
    "run_meta",
}
V51_PREDICTION_DESCRIPTOR_FIELDS = {"path", "sha256"}
SOFTWARE_FIELDS = {
    "cell_eval2_git_commit",
    "cell_eval2_checkout_clean",
    "cell_eval2_declared_version",
    "cell_eval2_runtime_version",
    "cell_eval2_module",
    "pdex_version",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def _load_authenticated_json(
    path: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and hash the exact receipt bytes under a stable file identity."""

    canonical = path.resolve()
    require(canonical.is_file(), f"{label} is missing: {canonical}")
    before = canonical.stat()
    try:
        raw = canonical.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {canonical}") from error
    after = canonical.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"{label} changed while being authenticated",
    )
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload, {
        "path": str(canonical),
        "size_bytes": after.st_size,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _authenticate_descriptor(
    record: Any,
    expected_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    """Re-hash one closed descriptor while detecting concurrent mutation."""

    require(isinstance(record, dict), f"{label}: descriptor is not an object")
    require(
        set(record) in (DESCRIPTOR_BASE_FIELDS, DESCRIPTOR_WITH_MTIME_FIELDS),
        f"{label}: descriptor fields are not recognized",
    )
    canonical = expected_path.resolve()
    require(record.get("path") == str(canonical), f"{label}: descriptor path mismatch")
    size = record.get("size_bytes")
    digest = record.get("sha256")
    require(type(size) is int and size >= 0, f"{label}: invalid size stamp")
    require(
        isinstance(digest, str) and SHA256_PATTERN.fullmatch(digest) is not None,
        f"{label}: invalid SHA-256 stamp",
    )
    require(canonical.is_file(), f"{label}: stamped file is missing")
    before = canonical.stat()
    observed_digest = _sha256_file(canonical)
    after = canonical.stat()
    require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        f"{label}: file changed while being authenticated",
    )
    require(after.st_size == size, f"{label}: stamped size mismatch")
    require(observed_digest == digest, f"{label}: stamped SHA-256 mismatch")
    if "mtime_ns" in record:
        require(
            type(record["mtime_ns"]) is int
            and record["mtime_ns"] == after.st_mtime_ns,
            f"{label}: stamped modification time mismatch",
        )
    return {
        "path": str(canonical),
        "size_bytes": int(size),
        "sha256": str(digest),
    }


def _require_same_descriptor(left: Any, right: Any, label: str) -> None:
    require(isinstance(left, dict) and isinstance(right, dict), f"{label}: descriptor is absent")
    for field in ("path", "size_bytes", "sha256"):
        require(left.get(field) == right.get(field), f"{label}: descriptor differs in {field}")
    if "mtime_ns" in left or "mtime_ns" in right:
        require(left.get("mtime_ns") == right.get("mtime_ns"), f"{label}: descriptor differs in mtime_ns")


def _validate_software(record: Any, label: str) -> dict[str, str]:
    require(isinstance(record, dict), f"{label}: software identity is not an object")
    require(set(record) == SOFTWARE_FIELDS, f"{label}: software fields are not exact")
    require(
        record["cell_eval2_git_commit"] == EXPECTED_CELL_EVAL_COMMIT,
        f"{label}: cell-eval2 commit mismatch",
    )
    require(record["cell_eval2_checkout_clean"] is True, f"{label}: scorer was dirty")
    require(
        record["cell_eval2_declared_version"] == EXPECTED_CELL_EVAL_VERSION,
        f"{label}: cell-eval2 version mismatch",
    )
    require(record["pdex_version"] == EXPECTED_PDEX_VERSION, f"{label}: pdex mismatch")
    runtime = record["cell_eval2_runtime_version"]
    module = record["cell_eval2_module"]
    require(isinstance(runtime, str) and bool(runtime), f"{label}: runtime version is absent")
    require(isinstance(module, str) and bool(module), f"{label}: module path is absent")
    return {
        "cell_eval2_git_commit": EXPECTED_CELL_EVAL_COMMIT,
        "cell_eval2_declared_version": EXPECTED_CELL_EVAL_VERSION,
        "cell_eval2_runtime_version": runtime,
        "pdex_version": EXPECTED_PDEX_VERSION,
    }


def _validate_run_meta(
    path: Path,
    *,
    resolved: dict[str, Any],
    runtime_version: str,
    label: str,
) -> dict[str, str]:
    run = _load_json(path, f"{label}: run metadata")
    require(run.get("config_digest") == resolved.get("config_digest"), f"{label}: config digest mismatch")
    require(run.get("resolved_device") == "cpu", f"{label}: scorer did not use CPU")
    require(run.get("resolved_de_backend") == "pdex", f"{label}: backend mismatch")
    require(run.get("comparator") == "bulk_lognorm", f"{label}: comparator mismatch")
    require(run.get("source_fingerprint_strict") is True, f"{label}: strict source fingerprinting is disabled")
    require(
        run.get("input_type_real_effective") == "counts"
        and run.get("input_type_pred_effective") == "counts",
        f"{label}: scorer inputs are not counts",
    )
    if "cell_eval2_version" in resolved:
        require(
            resolved["cell_eval2_version"] == runtime_version
            and run.get("cell_eval2_version") == runtime_version,
            f"{label}: runtime scorer version mismatch",
        )
    pdex = ((run.get("environment") or {}).get("pdex") or {}).get("version")
    require(pdex == EXPECTED_PDEX_VERSION, f"{label}: run metadata pdex mismatch")
    source_fingerprint = run.get("source_fingerprint")
    anchor_identity = run.get("anchor_semantic_identity")
    for name, value in (
        ("source fingerprint", source_fingerprint),
        ("anchor semantic identity", anchor_identity),
        ("config digest", resolved.get("config_digest")),
    ):
        require(
            isinstance(value, str) and SHA256_PATTERN.fullmatch(value) is not None,
            f"{label}: {name} is malformed",
        )
    return {
        "config_digest": str(resolved["config_digest"]),
        "source_fingerprint": str(source_fingerprint),
        "anchor_semantic_identity": str(anchor_identity),
    }


def _authenticate_output_files(
    outputs: dict[str, Any],
    results_path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    root = results_path.resolve().parent
    expected = {
        "results": results_path,
        "aggregate_results": root / "agg_results.csv",
        "metric_aggregation": root / "metric_aggregation.csv",
        "run_meta": root / "run_meta.json",
    }
    return {
        name: _authenticate_descriptor(outputs.get(name), path, label=f"{label}|{name}")
        for name, path in expected.items()
    }


def _authenticate_v6(
    receipt_path: Path,
    receipt: dict[str, Any],
    results_path: Path,
    candidate_spec_path: Path | None,
    *,
    context: str,
    label: str,
    expected_targets: int,
    allow_legacy_v1: bool,
    manifest_path: Path,
) -> dict[str, Any]:
    prefix = f"{context}|{label}|V6"
    require(candidate_spec_path is not None, f"{prefix}: candidate spec is required")
    require(set(receipt) == V6_TOP_LEVEL_FIELDS, f"{prefix}: receipt fields are not exact")
    require(receipt.get("status") == "passed", f"{prefix}: scoring did not pass")
    require(receipt.get("context") == context, f"{prefix}: receipt context mismatch")
    require(receipt.get("output_tag") == label, f"{prefix}: label does not match output tag")
    software = _validate_software(receipt.get("software"), prefix)
    resolved = receipt.get("resolved_run")
    require(isinstance(resolved, dict), f"{prefix}: resolved run is absent")
    require(resolved.get("resolved_device") == "cpu", f"{prefix}: device mismatch")
    require(resolved.get("resolved_de_backend") == "pdex", f"{prefix}: backend mismatch")
    require(resolved.get("comparator") == "bulk_lognorm", f"{prefix}: comparator mismatch")
    require(resolved.get("source_fingerprint_strict") is True, f"{prefix}: strict fingerprinting is disabled")
    require(resolved.get("pdex_version") == EXPECTED_PDEX_VERSION, f"{prefix}: resolved pdex mismatch")
    contract = receipt.get("contract")
    require(
        isinstance(contract, dict)
        and set(contract) == V6_CONTRACT_FIELDS
        and all(value is True for value in contract.values()),
        f"{prefix}: scorer aggregation contract is incomplete",
    )
    outputs = receipt.get("outputs")
    require(isinstance(outputs, dict) and set(outputs) == V6_OUTPUT_FIELDS, f"{prefix}: output fields are not exact")
    require(outputs.get("expected_targets") == expected_targets, f"{prefix}: target count mismatch")
    authenticated_outputs = _authenticate_output_files(outputs, results_path, label=prefix)

    candidate_receipts = receipt.get("candidate_receipts")
    require(
        isinstance(candidate_receipts, dict)
        and set(candidate_receipts)
        in (V6_CANDIDATE_RECEIPT_BASE_FIELDS, V6_CANDIDATE_RECEIPT_STRICT_FIELDS),
        f"{prefix}: candidate receipt fields are not exact",
    )
    spec_path = candidate_spec_path.resolve()
    generation_verification_path = spec_path.parent / "generation_verified.json"
    views_path = receipt_path.resolve().parent / "views.json"
    authenticated_candidate_receipts = {
        "spec": _authenticate_descriptor(candidate_receipts["spec"], spec_path, label=f"{prefix}|spec"),
        "generation_verification": _authenticate_descriptor(
            candidate_receipts["generation_verification"],
            generation_verification_path,
            label=f"{prefix}|generation_verification",
        ),
        "views": _authenticate_descriptor(candidate_receipts["views"], views_path, label=f"{prefix}|views"),
    }
    spec = _load_json(spec_path, f"{prefix}: candidate spec")
    spec_schema = spec.get("schema")
    require(
        spec_schema == V6_SPEC_SCHEMA
        or (spec_schema == LEGACY_V6_SPEC_SCHEMA and allow_legacy_v1),
        f"{prefix}: legacy V6 spec requires explicit read-only permission",
    )
    if spec_schema == V6_SPEC_SCHEMA:
        require(
            set(candidate_receipts) == V6_CANDIDATE_RECEIPT_STRICT_FIELDS,
            f"{prefix}: strict v2 receipt lacks authenticated scoring views",
        )
    require(spec.get("context") == context, f"{prefix}: candidate spec context mismatch")
    require(spec.get("output_tag") == label, f"{prefix}: candidate spec tag mismatch")
    configuration = spec.get("configuration")
    require(isinstance(configuration, dict), f"{prefix}: candidate configuration is absent")
    require(configuration.get("context") == context, f"{prefix}: candidate configuration context mismatch")
    if spec_schema == V6_SPEC_SCHEMA:
        _validate_strict_spec_provenance(spec)
    inputs = spec.get("inputs")
    require(isinstance(inputs, dict), f"{prefix}: candidate spec inputs are absent")
    panel_descriptor = _authenticate_descriptor(
        inputs.get("panel_csv"),
        manifest_path,
        label=f"{prefix}|panel_csv",
    )

    generation_verification = _load_json(
        generation_verification_path, f"{prefix}: generation verification"
    )
    require(
        generation_verification.get("schema") == V6_GENERATION_VERIFICATION_SCHEMA
        and generation_verification.get("status") == "passed",
        f"{prefix}: generation verification did not pass",
    )
    require(
        generation_verification.get("context") == context
        and generation_verification.get("output_tag") == label,
        f"{prefix}: generation verification identity mismatch",
    )
    checks = generation_verification.get("checks")
    require(
        isinstance(checks, dict)
        and bool(checks)
        and all(value is True for value in checks.values()),
        f"{prefix}: generation verification checks are incomplete",
    )
    if spec_schema == V6_SPEC_SCHEMA:
        require(
            checks.get("strict_generator_provenance_authenticated") is True,
            f"{prefix}: strict generator provenance was not authenticated",
        )
    require(
        generation_verification.get("configuration") == configuration,
        f"{prefix}: generation and planned configurations differ",
    )
    generation_provenance = generation_verification.get("provenance")
    require(isinstance(generation_provenance, dict), f"{prefix}: generation provenance is absent")
    _require_same_descriptor(
        generation_provenance.get("spec"),
        candidate_receipts.get("spec"),
        f"{prefix}|generation_spec",
    )
    views = _load_json(views_path, f"{prefix}: scoring views")
    require(
        set(views) == VIEWS_FIELDS
        and views.get("schema") == "vcc-public-scoring-views-v1"
        and views.get("context") == context,
        f"{prefix}: scoring views identity mismatch",
    )
    views_provenance = views.get("provenance")
    require(
        isinstance(views_provenance, dict)
        and set(views_provenance) == VIEWS_PROVENANCE_FIELDS,
        f"{prefix}: scoring views provenance is incomplete",
    )
    _require_same_descriptor(
        views_provenance.get("prediction"),
        generation_provenance.get("prediction_h5ad"),
        f"{prefix}|views_prediction",
    )
    expected_prediction_view = (receipt_path.resolve().parent / "prediction_view.h5ad").resolve()
    expected_truth_view = (receipt_path.resolve().parent / "truth_view.h5ad").resolve()
    require(
        Path(str((views_provenance.get("output_prediction") or {}).get("path", ""))).resolve()
        == expected_prediction_view,
        f"{prefix}: prediction-view path mismatch",
    )
    require(
        Path(str((views_provenance.get("output_truth") or {}).get("path", ""))).resolve()
        == expected_truth_view,
        f"{prefix}: truth-view path mismatch",
    )
    authenticated_view_outputs = {
        "prediction_view": _authenticate_descriptor(
            views_provenance.get("output_prediction"),
            expected_prediction_view,
            label=f"{prefix}|prediction_view",
        ),
        "truth_view": _authenticate_descriptor(
            views_provenance.get("output_truth"),
            expected_truth_view,
            label=f"{prefix}|truth_view",
        ),
    }
    if set(candidate_receipts) == V6_CANDIDATE_RECEIPT_STRICT_FIELDS:
        for name, expected in authenticated_view_outputs.items():
            _authenticate_descriptor(
                candidate_receipts[name],
                Path(expected["path"]),
                label=f"{prefix}|receipt_{name}",
            )
    run_identity = _validate_run_meta(
        Path(authenticated_outputs["run_meta"]["path"]),
        resolved=resolved,
        runtime_version=software["cell_eval2_runtime_version"],
        label=prefix,
    )
    run_metadata = _load_json(
        Path(authenticated_outputs["run_meta"]["path"]), f"{prefix}: run metadata"
    )
    require(
        Path(str(run_metadata.get("source", ""))).resolve() == expected_truth_view,
        f"{prefix}: run metadata truth source mismatch",
    )
    return {
        "kind": "v6",
        "context": context,
        "label": label,
        "receipt": str(receipt_path.resolve()),
        "candidate_spec": authenticated_candidate_receipts["spec"],
        "candidate_configuration": configuration,
        "candidate_spec_mode": (
            "strict-v2" if spec_schema == V6_SPEC_SCHEMA else "legacy-v1-read-only"
        ),
        "outputs": authenticated_outputs,
        "panel_csv": panel_descriptor,
        "view_outputs": authenticated_view_outputs,
        "software": software,
        **run_identity,
    }


def _authenticate_v51(
    receipt_path: Path,
    receipt: dict[str, Any],
    results_path: Path,
    receipt_role: str | None,
    *,
    context: str,
    label: str,
) -> dict[str, Any]:
    prefix = f"{context}|{label}|V5.1"
    require(receipt_role in V51_ROLES, f"{prefix}: an exact V5.1 receipt role is required")
    require(label == receipt_role, f"{prefix}: label does not match V5.1 receipt role")
    require(set(receipt) == V51_TOP_LEVEL_FIELDS, f"{prefix}: receipt fields are not exact")
    require(receipt.get("status") == "scoring_complete", f"{prefix}: scoring is incomplete")
    require(receipt.get("context") == context, f"{prefix}: receipt context mismatch")
    software = _validate_software(receipt.get("scoring_software"), prefix)
    resolved_all = receipt.get("resolved_scoring")
    outputs_all = receipt.get("outputs")
    require(
        isinstance(resolved_all, dict) and set(resolved_all) == set(V51_ROLES),
        f"{prefix}: resolved scorer roles are not exact",
    )
    require(
        isinstance(outputs_all, dict) and set(outputs_all) == set(V51_ROLES),
        f"{prefix}: output roles are not exact",
    )
    predictions_all = receipt.get("predictions")
    require(
        isinstance(predictions_all, dict) and set(predictions_all) == set(V51_ROLES),
        f"{prefix}: prediction roles are not exact",
    )
    source_prediction_record = predictions_all[receipt_role]
    require(
        isinstance(source_prediction_record, dict)
        and set(source_prediction_record) == V51_PREDICTION_DESCRIPTOR_FIELDS
        and isinstance(source_prediction_record.get("path"), str)
        and isinstance(source_prediction_record.get("sha256"), str)
        and SHA256_PATTERN.fullmatch(source_prediction_record["sha256"]) is not None,
        f"{prefix}: source prediction descriptor is malformed",
    )
    require(
        len({record.get("config_digest") for record in resolved_all.values()}) == 1,
        f"{prefix}: V5.1 scorer configurations differ",
    )
    resolved = resolved_all[receipt_role]
    require(isinstance(resolved, dict), f"{prefix}: resolved role is absent")
    require(resolved.get("resolved_device") == "cpu", f"{prefix}: device mismatch")
    require(resolved.get("resolved_de_backend") == "pdex", f"{prefix}: backend mismatch")
    require(resolved.get("pdex_version") == EXPECTED_PDEX_VERSION, f"{prefix}: resolved pdex mismatch")
    output_record = outputs_all[receipt_role]
    require(
        isinstance(output_record, dict) and set(output_record) == V51_OUTPUT_FIELDS,
        f"{prefix}: output fields are not exact",
    )
    authenticated_outputs = _authenticate_output_files(output_record, results_path, label=prefix)
    authenticated_outputs["views_report"] = _authenticate_descriptor(
        output_record["views_report"],
        receipt_path.resolve().parent / f"{receipt_role}_views.json",
        label=f"{prefix}|views_report",
    )
    views_path = Path(authenticated_outputs["views_report"]["path"])
    views = _load_json(views_path, f"{prefix}: scoring views")
    require(
        set(views) == VIEWS_FIELDS
        and views.get("schema") == "vcc-public-scoring-views-v1"
        and views.get("context") == context,
        f"{prefix}: scoring views identity mismatch",
    )
    views_contract = views.get("contract")
    require(
        isinstance(views_contract, dict)
        and views_contract.get("model_read_treated_truth") is False
        and views_contract.get("input_type") == "raw-nonnegative-integer-counts",
        f"{prefix}: scoring views firewall differs",
    )
    views_provenance = views.get("provenance")
    require(
        isinstance(views_provenance, dict)
        and set(views_provenance) == VIEWS_PROVENANCE_FIELDS,
        f"{prefix}: scoring views provenance is incomplete",
    )
    source_prediction_path = Path(source_prediction_record["path"]).resolve()
    authenticated_source_prediction = _authenticate_descriptor(
        views_provenance.get("prediction"),
        source_prediction_path,
        label=f"{prefix}|source_prediction",
    )
    require(
        authenticated_source_prediction["sha256"]
        == source_prediction_record["sha256"],
        f"{prefix}: scoring-view source prediction differs from the receipt role",
    )
    expected_prediction_view = (
        receipt_path.resolve().parent / f"{receipt_role}_prediction_view.h5ad"
    )
    expected_truth_view = (
        receipt_path.resolve().parent / f"{receipt_role}_truth_view.h5ad"
    )
    authenticated_view_outputs = {
        "prediction_view": _authenticate_descriptor(
            views_provenance.get("output_prediction"),
            expected_prediction_view,
            label=f"{prefix}|prediction_view",
        ),
        "truth_view": _authenticate_descriptor(
            views_provenance.get("output_truth"),
            expected_truth_view,
            label=f"{prefix}|truth_view",
        ),
    }
    run_identity = _validate_run_meta(
        Path(authenticated_outputs["run_meta"]["path"]),
        resolved=resolved,
        runtime_version=software["cell_eval2_runtime_version"],
        label=prefix,
    )
    run_metadata = _load_json(
        Path(authenticated_outputs["run_meta"]["path"]), f"{prefix}: run metadata"
    )
    require(
        Path(str(run_metadata.get("source", ""))).resolve()
        == expected_truth_view.resolve(),
        f"{prefix}: run metadata truth source mismatch",
    )
    return {
        "kind": "v51",
        "context": context,
        "label": label,
        "receipt_role": receipt_role,
        "receipt": str(receipt_path.resolve()),
        "candidate_spec": None,
        "candidate_configuration": {
            "target_remaining_fraction": 0.20,
            "residual_alpha": 0.0 if receipt_role == "anchor" else 0.10,
        },
        "outputs": authenticated_outputs,
        "source_prediction": authenticated_source_prediction,
        "view_outputs": authenticated_view_outputs,
        "software": software,
        **run_identity,
    }


def authenticate_scoring_bundle(
    *,
    receipt_path: Path,
    results_path: Path,
    context: str,
    label: str,
    expected_targets: int,
    receipt_role: str | None = None,
    candidate_spec_path: Path | None = None,
    allow_legacy_v1: bool = False,
    manifest_path: Path,
) -> dict[str, Any]:
    """Authenticate a V5.1 role or one V6 candidate and return its identity."""

    require(expected_targets > 0, "Expected target count must be positive")
    receipt, receipt_descriptor = _load_authenticated_json(
        receipt_path.resolve(), f"{context}|{label}: scoring receipt"
    )
    schema = receipt.get("schema")
    if schema == V6_RECEIPT_SCHEMA:
        require(receipt_role is None, f"{context}|{label}: V6 cannot use a V5.1 role")
        authenticated = _authenticate_v6(
            receipt_path,
            receipt,
            results_path,
            candidate_spec_path,
            context=context,
            label=label,
            expected_targets=expected_targets,
            allow_legacy_v1=allow_legacy_v1,
            manifest_path=manifest_path,
        )
    elif schema == V51_RECEIPT_SCHEMA:
        require(candidate_spec_path is None, f"{context}|{label}: V5.1 has no V6 candidate spec")
        authenticated = _authenticate_v51(
            receipt_path,
            receipt,
            results_path,
            receipt_role,
            context=context,
            label=label,
        )
    else:
        raise RuntimeError(f"{context}|{label}: unknown scoring receipt schema")
    authenticated["receipt"] = receipt_descriptor
    return authenticated


def require_shared_scoring_identity(records: dict[str, dict[str, Any]]) -> dict[str, str]:
    """Require all compared bundles to use the same scorer and sealed truth."""

    require(bool(records), "At least one authenticated scoring bundle is required")
    fields = (
        "config_digest",
        "source_fingerprint",
        "anchor_semantic_identity",
    )
    software_fields = (
        "cell_eval2_git_commit",
        "cell_eval2_declared_version",
        "cell_eval2_runtime_version",
        "pdex_version",
    )
    for field in fields:
        require(
            len({str(record[field]) for record in records.values()}) == 1,
            f"Compared scorer bundles differ in {field}",
        )
    for field in software_fields:
        require(
            len({str(record["software"][field]) for record in records.values()}) == 1,
            f"Compared scorer bundles differ in {field}",
        )
    first = next(iter(records.values()))
    return {
        **{field: str(first[field]) for field in fields},
        **{field: str(first["software"][field]) for field in software_fields},
    }
