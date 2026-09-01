#!/usr/bin/env python3
"""Authenticate completed V5.1 scorer receipts before summary consumption.

The scorer writes one receipt per public context only after both the fixed
STATE anchor and the V5.1 candidate have completed.  This validator treats the
receipt as a closed schema, re-hashes every stamped scorer output, and checks
that the two runs share the pinned scoring identity.  It deliberately does not
rewrite either receipts or scorer outputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any


SCHEMA = "vcc-public-validation-scoring-receipt-v1"
REPORT_SCHEMA = "vcc-public-v51-receipt-validation-v1"
FULL_SUMMARY_SCHEMA = "vcc-public-state-anchor-full-v1"
VIEWS_SCHEMA = "vcc-public-scoring-views-v1"
EXPECTED_CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
EXPECTED_CELL_EVAL_VERSION = "0.16.0"
EXPECTED_PDEX_VERSION = "0.3.0"
EXPECTED_CONTEXTS = {"HepG2": "hepg2", "Jurkat": "jurkat"}
EXPECTED_LABELS = ("anchor", "v51_alpha010")

TOP_LEVEL_FIELDS = {
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
SCORING_SOFTWARE_FIELDS = {
    "cell_eval2_git_commit",
    "cell_eval2_checkout_clean",
    "cell_eval2_declared_version",
    "cell_eval2_runtime_version",
    "cell_eval2_module",
    "pdex_version",
}
RESOLVED_SCORING_FIELDS = {
    "config_digest",
    "resolved_device",
    "resolved_de_backend",
    "cell_eval2_version",
    "pdex_version",
}
OUTPUT_FIELDS = {
    "views_report",
    "results",
    "aggregate_results",
    "metric_aggregation",
    "run_meta",
}
DESCRIPTOR_FIELDS = {"path", "size_bytes", "sha256"}
HASH_DESCRIPTOR_FIELDS = {"path", "sha256"}
SOURCE_DESCRIPTOR_FIELDS = {"path", "mtime_ns", "size_bytes", "sha256"}
GENERATION_QC_FIELDS = {
    "clip_fraction",
    "direct_changed_entries",
    "fallback_changed_entries",
    "anchor_sha256",
    "candidate_sha256",
}
VIEWS_FIELDS = {"schema", "context", "axes", "contract", "provenance"}
VIEWS_AXES_FIELDS = {
    "genes",
    "observed_control_cells",
    "prediction_treated_cells",
    "prediction_view_cells",
    "targets",
    "truth_view_cells",
}
VIEWS_CONTRACT = {
    "frozen_common_axis_authenticated": True,
    "gene_axis": "frozen-state-hepg2-jurkat-intersection",
    "input_type": "raw-nonnegative-integer-counts",
    "model_read_treated_truth": False,
    "prediction_controls_match_sealed_truth_exactly": True,
    "prediction_controls_source": "released-controls-only-input",
}
VIEWS_PROVENANCE_FIELDS = {
    "common_genes",
    "common_genes_json",
    "controls_only",
    "output_prediction",
    "output_truth",
    "prediction",
    "sealed_truth",
}
OUTPUT_LAYOUT = {
    "views_report": "{label}_views.json",
    "results": "{label}_cell_eval2/results.csv",
    "aggregate_results": "{label}_cell_eval2/agg_results.csv",
    "metric_aggregation": "{label}_cell_eval2/metric_aggregation.csv",
    "run_meta": "{label}_cell_eval2/run_meta.json",
}
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_identity(
    path: Path,
    hash_cache: dict[Path, tuple[int, int, str]],
    *,
    label: str,
) -> tuple[int, int, str]:
    """Return a stable size/mtime/hash identity, hashing each path once."""
    require(path.is_file(), f"{label}: stamped file is missing")
    stat_before = path.stat()
    cached = hash_cache.get(path)
    if cached is not None:
        require(
            cached[:2] == (stat_before.st_size, stat_before.st_mtime_ns),
            f"{label}: stamped file changed during validation",
        )
        return cached
    digest = sha256_file(path)
    stat_after = path.stat()
    require(
        (stat_before.st_size, stat_before.st_mtime_ns)
        == (stat_after.st_size, stat_after.st_mtime_ns),
        f"{label}: stamped file changed while being hashed",
    )
    identity = (stat_after.st_size, stat_after.st_mtime_ns, digest)
    hash_cache[path] = identity
    return identity


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file(), f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not valid JSON: {path}") from error
    require(isinstance(payload, dict), f"{label} must be a JSON object")
    return payload


def validate_file_descriptor(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    hash_cache: dict[Path, tuple[int, int, str]] | None = None,
) -> dict[str, Any]:
    """Validate one closed file descriptor and re-hash its exact path."""
    require(isinstance(record, dict), f"{label}: descriptor is not an object")
    require(
        set(record) == DESCRIPTOR_FIELDS,
        f"{label}: descriptor fields differ from {sorted(DESCRIPTOR_FIELDS)}",
    )
    canonical_path = expected_path.resolve()
    require(
        record["path"] == str(canonical_path),
        f"{label}: descriptor path mismatch",
    )
    size = record["size_bytes"]
    require(
        type(size) is int and size >= 0,
        f"{label}: descriptor size_bytes is not a nonnegative integer",
    )
    recorded_hash = record["sha256"]
    require(
        isinstance(recorded_hash, str)
        and SHA256_PATTERN.fullmatch(recorded_hash) is not None,
        f"{label}: descriptor SHA-256 is malformed",
    )
    if hash_cache is None:
        hash_cache = {}
    actual_size, _, actual_hash = _file_identity(
        canonical_path, hash_cache, label=label
    )
    require(
        actual_size == size,
        f"{label}: stamped file size mismatch",
    )
    require(
        actual_hash == recorded_hash,
        f"{label}: stamped file hash mismatch",
    )
    return {
        "path": str(canonical_path),
        "size_bytes": size,
        "sha256": recorded_hash,
    }


def _validate_hash_descriptor(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    hash_cache: dict[Path, tuple[int, int, str]],
) -> dict[str, Any]:
    """Validate and re-hash a receipt descriptor that predates size stamps."""
    require(isinstance(record, dict), f"{label}: descriptor is not an object")
    require(
        set(record) == HASH_DESCRIPTOR_FIELDS,
        f"{label}: descriptor fields differ from {sorted(HASH_DESCRIPTOR_FIELDS)}",
    )
    canonical_path = expected_path.resolve()
    require(record["path"] == str(canonical_path), f"{label}: descriptor path mismatch")
    recorded_hash = record["sha256"]
    require(
        isinstance(recorded_hash, str)
        and SHA256_PATTERN.fullmatch(recorded_hash) is not None,
        f"{label}: descriptor SHA-256 is malformed",
    )
    actual_size, _, actual_hash = _file_identity(
        canonical_path, hash_cache, label=label
    )
    require(actual_hash == recorded_hash, f"{label}: stamped file hash mismatch")
    return {
        "path": str(canonical_path),
        "size_bytes": actual_size,
        "sha256": recorded_hash,
    }


def _validate_source_descriptor(
    record: Any,
    expected_path: Path,
    *,
    label: str,
    hash_cache: dict[Path, tuple[int, int, str]],
) -> dict[str, Any]:
    """Validate a view-input descriptor, including its recorded modification time."""
    require(isinstance(record, dict), f"{label}: descriptor is not an object")
    require(
        set(record) == SOURCE_DESCRIPTOR_FIELDS,
        f"{label}: descriptor fields differ from {sorted(SOURCE_DESCRIPTOR_FIELDS)}",
    )
    canonical_path = expected_path.resolve()
    require(record["path"] == str(canonical_path), f"{label}: descriptor path mismatch")
    size = record["size_bytes"]
    mtime_ns = record["mtime_ns"]
    recorded_hash = record["sha256"]
    require(
        type(size) is int and size >= 0,
        f"{label}: descriptor size_bytes is not a nonnegative integer",
    )
    require(
        type(mtime_ns) is int and mtime_ns >= 0,
        f"{label}: descriptor mtime_ns is not a nonnegative integer",
    )
    require(
        isinstance(recorded_hash, str)
        and SHA256_PATTERN.fullmatch(recorded_hash) is not None,
        f"{label}: descriptor SHA-256 is malformed",
    )
    actual_size, actual_mtime_ns, actual_hash = _file_identity(
        canonical_path, hash_cache, label=label
    )
    require(actual_size == size, f"{label}: stamped file size mismatch")
    require(actual_mtime_ns == mtime_ns, f"{label}: stamped file mtime mismatch")
    require(actual_hash == recorded_hash, f"{label}: stamped file hash mismatch")
    return {
        "path": str(canonical_path),
        "mtime_ns": mtime_ns,
        "size_bytes": size,
        "sha256": recorded_hash,
    }


def _validate_generation_qc(record: Any, *, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label}: generation QC is not an object")
    require(set(record) == GENERATION_QC_FIELDS, f"{label}: generation QC fields differ")
    clip_fraction = record["clip_fraction"]
    direct_changed = record["direct_changed_entries"]
    fallback_changed = record["fallback_changed_entries"]
    require(
        type(clip_fraction) in (int, float)
        and math.isfinite(float(clip_fraction))
        and 0.0 <= float(clip_fraction) < 1e-4,
        f"{label}: invalid generation clip fraction",
    )
    require(
        type(direct_changed) is int and direct_changed > 0,
        f"{label}: invalid direct-change count",
    )
    require(
        type(fallback_changed) is int and fallback_changed == 0,
        f"{label}: held-target output is not identical",
    )
    for hash_field in ("anchor_sha256", "candidate_sha256"):
        require(
            isinstance(record[hash_field], str)
            and SHA256_PATTERN.fullmatch(record[hash_field]) is not None,
            f"{label}: malformed {hash_field}",
        )
    return dict(record)


def _validate_full_generation_summary(
    record: Any,
    *,
    project_root: Path,
    context: str,
    hash_cache: dict[Path, tuple[int, int, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = (
        project_root
        / "artifacts/public_v51/full/public_state_anchor_v51_full_summary.json"
    )
    descriptor = _validate_hash_descriptor(
        record,
        expected_path,
        label=f"{context}: full-generation summary",
        hash_cache=hash_cache,
    )
    summary = _load_json_object(expected_path, f"{context}: full-generation summary")
    require(
        set(summary) == {"schema", "contexts"}
        and summary["schema"] == FULL_SUMMARY_SCHEMA,
        f"{context}: full-generation summary identity mismatch",
    )
    contexts = summary["contexts"]
    require(
        isinstance(contexts, dict) and set(contexts) == set(EXPECTED_CONTEXTS),
        f"{context}: full-generation context set is not exact",
    )
    validated_contexts = {
        name: _validate_generation_qc(
            contexts[name], label=f"{context}: full-generation summary {name}"
        )
        for name in EXPECTED_CONTEXTS
    }
    return descriptor, validated_contexts[context]


def _validate_scoring_software(
    record: Any,
    *,
    project_root: Path,
    context: str,
) -> dict[str, str]:
    require(isinstance(record, dict), f"{context}: scoring_software is not an object")
    require(
        set(record) == SCORING_SOFTWARE_FIELDS,
        f"{context}: scoring_software fields are not exact",
    )
    require(
        record["cell_eval2_git_commit"] == EXPECTED_CELL_EVAL_COMMIT,
        f"{context}: cell-eval2 scoring revision mismatch",
    )
    require(
        record["cell_eval2_checkout_clean"] is True,
        f"{context}: scorer checkout was not stamped clean",
    )
    require(
        record["cell_eval2_declared_version"] == EXPECTED_CELL_EVAL_VERSION,
        f"{context}: cell-eval2 declared version mismatch",
    )
    require(
        record["pdex_version"] == EXPECTED_PDEX_VERSION,
        f"{context}: pdex version mismatch",
    )
    runtime_version = record["cell_eval2_runtime_version"]
    require(
        isinstance(runtime_version, str) and bool(runtime_version),
        f"{context}: cell-eval2 runtime version is absent",
    )
    expected_module = (
        project_root / "external/cell-eval2/src/cell_eval2/__init__.py"
    ).resolve()
    require(
        record["cell_eval2_module"] == str(expected_module),
        f"{context}: cell-eval2 module path mismatch",
    )
    return {
        "cell_eval2_git_commit": EXPECTED_CELL_EVAL_COMMIT,
        "cell_eval2_declared_version": EXPECTED_CELL_EVAL_VERSION,
        "cell_eval2_runtime_version": runtime_version,
        "pdex_version": EXPECTED_PDEX_VERSION,
    }


def _validate_resolved_record(
    record: Any,
    *,
    context: str,
    label: str,
    runtime_version: str,
) -> dict[str, str]:
    prefix = f"{context}|{label}"
    require(isinstance(record, dict), f"{prefix}: resolved scorer record is not an object")
    require(
        set(record) == RESOLVED_SCORING_FIELDS,
        f"{prefix}: resolved scorer fields are not exact",
    )
    config_digest = record["config_digest"]
    require(
        isinstance(config_digest, str)
        and SHA256_PATTERN.fullmatch(config_digest) is not None,
        f"{prefix}: scorer config digest is malformed",
    )
    require(record["resolved_device"] == "cpu", f"{prefix}: scorer did not use CPU")
    require(
        record["resolved_de_backend"] == "pdex",
        f"{prefix}: scorer backend mismatch",
    )
    require(
        record["pdex_version"] == EXPECTED_PDEX_VERSION,
        f"{prefix}: resolved pdex version mismatch",
    )
    require(
        record["cell_eval2_version"] == runtime_version,
        f"{prefix}: runtime scorer version mismatch",
    )
    return {
        "config_digest": config_digest,
        "cell_eval2_version": runtime_version,
        "pdex_version": EXPECTED_PDEX_VERSION,
        "resolved_de_backend": "pdex",
        "resolved_device": "cpu",
    }


def _validate_run_meta(
    path: Path,
    resolved: dict[str, str],
    *,
    expected_source: Path,
    context: str,
    label: str,
) -> dict[str, str]:
    prefix = f"{context}|{label}"
    metadata = _load_json_object(path, f"{prefix}: run_meta")
    for field in (
        "config_digest",
        "cell_eval2_version",
        "resolved_de_backend",
        "resolved_device",
    ):
        require(
            metadata.get(field) == resolved[field],
            f"{prefix}: run metadata {field} differs from the receipt",
        )
    pdex = ((metadata.get("environment") or {}).get("pdex") or {}).get("version")
    require(pdex == resolved["pdex_version"], f"{prefix}: run metadata pdex mismatch")
    require(
        metadata.get("source_fingerprint_strict") is True,
        f"{prefix}: strict source fingerprinting is disabled",
    )
    source_fingerprint = metadata.get("source_fingerprint")
    require(
        isinstance(source_fingerprint, str)
        and SHA256_PATTERN.fullmatch(source_fingerprint) is not None,
        f"{prefix}: source fingerprint is malformed",
    )
    require(
        metadata.get("source") == str(expected_source.resolve()),
        f"{prefix}: scorer source path mismatch",
    )
    require(metadata.get("comparator") == "bulk_lognorm", f"{prefix}: comparator mismatch")
    require(
        metadata.get("input_type_real_effective") == "counts"
        and metadata.get("input_type_pred_effective") == "counts",
        f"{prefix}: scorer input type mismatch",
    )
    anchor_identity = metadata.get("anchor_semantic_identity")
    require(
        isinstance(anchor_identity, str)
        and SHA256_PATTERN.fullmatch(anchor_identity) is not None,
        f"{prefix}: anchor semantic identity is malformed",
    )
    return {
        "anchor_semantic_identity": anchor_identity,
        "source_fingerprint": source_fingerprint,
    }


def _validate_views_report(
    path: Path,
    *,
    project_root: Path,
    context: str,
    context_slug: str,
    label: str,
    prediction: dict[str, Any],
    hash_cache: dict[Path, tuple[int, int, str]],
) -> dict[str, Any]:
    """Close the report-to-input/output provenance chain for one scoring view."""
    prefix = f"{context}|{label}"
    report = _load_json_object(path, f"{prefix}: views report")
    require(set(report) == VIEWS_FIELDS, f"{prefix}: views report fields are not exact")
    require(
        report["schema"] == VIEWS_SCHEMA and report["context"] == context,
        f"{prefix}: views report identity mismatch",
    )

    axes = report["axes"]
    require(
        isinstance(axes, dict) and set(axes) == VIEWS_AXES_FIELDS,
        f"{prefix}: views axes fields are not exact",
    )
    require(
        all(type(axes[field]) is int and axes[field] > 0 for field in VIEWS_AXES_FIELDS),
        f"{prefix}: views axes contain invalid counts",
    )
    require(
        axes["genes"] == 7107
        and axes["targets"] == 300
        and axes["prediction_treated_cells"] == 120000,
        f"{prefix}: frozen scoring axes differ",
    )
    require(
        axes["prediction_view_cells"]
        == axes["observed_control_cells"] + axes["prediction_treated_cells"],
        f"{prefix}: prediction-view cell count is inconsistent",
    )
    require(report["contract"] == VIEWS_CONTRACT, f"{prefix}: view contract differs")

    provenance = report["provenance"]
    require(
        isinstance(provenance, dict) and set(provenance) == VIEWS_PROVENANCE_FIELDS,
        f"{prefix}: views provenance fields are not exact",
    )
    dataset_root = project_root / "dataset/public_v51"
    expected_source_paths = {
        "common_genes": dataset_root / "common_genes.csv",
        "common_genes_json": dataset_root / "common_genes.json",
        "controls_only": dataset_root / f"{context_slug}_controls_only.h5ad",
        "sealed_truth": dataset_root / f"{context_slug}_sealed_truth.h5ad",
        "prediction": Path(prediction["path"]),
    }
    validated_sources = {
        source_name: _validate_source_descriptor(
            provenance[source_name],
            expected_source_path,
            label=f"{prefix}|{source_name}",
            hash_cache=hash_cache,
        )
        for source_name, expected_source_path in expected_source_paths.items()
    }
    require(
        validated_sources["prediction"]["sha256"] == prediction["sha256"],
        f"{prefix}: prediction provenance hash differs from the receipt",
    )

    prediction_view = path.parent / f"{label}_prediction_view.h5ad"
    truth_view = path.parent / f"{label}_truth_view.h5ad"
    validated_prediction_view = validate_file_descriptor(
        provenance["output_prediction"],
        prediction_view,
        label=f"{prefix}|output_prediction",
        hash_cache=hash_cache,
    )
    validated_truth_view = validate_file_descriptor(
        provenance["output_truth"],
        truth_view,
        label=f"{prefix}|output_truth",
        hash_cache=hash_cache,
    )
    return {
        "prediction": validated_sources["prediction"],
        "prediction_view": validated_prediction_view,
        "truth_view": validated_truth_view,
    }


def validate_scoring_receipt(receipt_path: Path, expected_context: str) -> dict[str, Any]:
    """Validate one completed context receipt without mutating any artifact."""
    require(expected_context in EXPECTED_CONTEXTS, f"Unknown context: {expected_context}")
    receipt_path = receipt_path.resolve()
    context_slug = EXPECTED_CONTEXTS[expected_context]
    require(len(receipt_path.parents) >= 5, f"{expected_context}: receipt path is too shallow")
    project_root = receipt_path.parents[4]
    expected_receipt_path = (
        project_root
        / "artifacts/public_v51/scoring"
        / context_slug
        / "scoring_receipt.json"
    ).resolve()
    require(
        receipt_path == expected_receipt_path,
        f"{expected_context}: receipt path is not the canonical context path",
    )
    receipt = _load_json_object(receipt_path, f"{expected_context}: scoring receipt")
    require(
        set(receipt) == TOP_LEVEL_FIELDS,
        f"{expected_context}: scoring receipt fields are not exact",
    )
    require(receipt["schema"] == SCHEMA, f"{expected_context}: bad scoring receipt schema")
    require(
        receipt["status"] == "scoring_complete",
        f"{expected_context}: scoring receipt is incomplete",
    )
    require(
        receipt["context"] == expected_context,
        f"{expected_context}: scoring receipt context mismatch",
    )
    hash_cache: dict[Path, tuple[int, int, str]] = {}
    full_summary, expected_generation_qc = _validate_full_generation_summary(
        receipt["full_generation_summary"],
        project_root=project_root,
        context=expected_context,
        hash_cache=hash_cache,
    )
    generation_qc = _validate_generation_qc(
        receipt["generation_qc"], label=f"{expected_context}: receipt"
    )
    require(
        generation_qc == expected_generation_qc,
        f"{expected_context}: receipt generation QC differs from the full summary",
    )

    predictions_record = receipt["predictions"]
    require(
        isinstance(predictions_record, dict)
        and set(predictions_record) == set(EXPECTED_LABELS),
        f"{expected_context}: prediction label set is not exact",
    )
    predictions: dict[str, dict[str, Any]] = {}
    for label in EXPECTED_LABELS:
        prediction_path = (
            project_root
            / "artifacts/public_v51/full"
            / f"{context_slug}_{label}.h5ad"
        )
        predictions[label] = _validate_hash_descriptor(
            predictions_record[label],
            prediction_path,
            label=f"{expected_context}|{label}|prediction",
            hash_cache=hash_cache,
        )
    require(
        predictions["anchor"]["sha256"] == generation_qc["anchor_sha256"],
        f"{expected_context}: anchor prediction hash differs from generation QC",
    )
    require(
        predictions["v51_alpha010"]["sha256"]
        == generation_qc["candidate_sha256"],
        f"{expected_context}: candidate prediction hash differs from generation QC",
    )

    software = _validate_scoring_software(
        receipt["scoring_software"],
        project_root=project_root,
        context=expected_context,
    )

    resolved_scoring = receipt["resolved_scoring"]
    require(
        isinstance(resolved_scoring, dict)
        and set(resolved_scoring) == set(EXPECTED_LABELS),
        f"{expected_context}: resolved scorer set is not exact",
    )
    resolved = {
        label: _validate_resolved_record(
            resolved_scoring[label],
            context=expected_context,
            label=label,
            runtime_version=software["cell_eval2_runtime_version"],
        )
        for label in EXPECTED_LABELS
    }
    require(
        len({record["config_digest"] for record in resolved.values()}) == 1,
        f"{expected_context}: anchor and candidate config identities differ",
    )

    outputs = receipt["outputs"]
    require(
        isinstance(outputs, dict) and set(outputs) == set(EXPECTED_LABELS),
        f"{expected_context}: scorer output label set is not exact",
    )
    validated_outputs: dict[str, dict[str, Any]] = {}
    run_identities: dict[str, dict[str, str]] = {}
    for label in EXPECTED_LABELS:
        output_record = outputs[label]
        prefix = f"{expected_context}|{label}"
        require(isinstance(output_record, dict), f"{prefix}: output record is not an object")
        require(set(output_record) == OUTPUT_FIELDS, f"{prefix}: output fields are not exact")
        validated_outputs[label] = {}
        for output_name, relative_template in OUTPUT_LAYOUT.items():
            expected_path = receipt_path.parent / relative_template.format(label=label)
            validated_outputs[label][output_name] = validate_file_descriptor(
                output_record[output_name],
                expected_path,
                label=f"{prefix}|{output_name}",
                hash_cache=hash_cache,
            )

        views_path = Path(validated_outputs[label]["views_report"]["path"])
        view_artifacts = _validate_views_report(
            views_path,
            project_root=project_root,
            context=expected_context,
            context_slug=context_slug,
            label=label,
            prediction=predictions[label],
            hash_cache=hash_cache,
        )
        expected_truth = receipt_path.parent / f"{label}_truth_view.h5ad"
        run_identities[label] = _validate_run_meta(
            Path(validated_outputs[label]["run_meta"]["path"]),
            resolved[label],
            expected_source=expected_truth,
            context=expected_context,
            label=label,
        )
        validated_outputs[label]["view_artifacts"] = view_artifacts

    require(
        len({record["source_fingerprint"] for record in run_identities.values()}) == 1,
        f"{expected_context}: anchor and candidate truth fingerprints differ",
    )
    require(
        len({record["anchor_semantic_identity"] for record in run_identities.values()}) == 1,
        f"{expected_context}: anchor semantic identities differ",
    )
    require(
        len(
            {
                record["view_artifacts"]["truth_view"]["sha256"]
                for record in validated_outputs.values()
            }
        )
        == 1,
        f"{expected_context}: anchor and candidate truth views differ",
    )
    shared_resolved = resolved[EXPECTED_LABELS[0]]
    return {
        "context": expected_context,
        "receipt": str(receipt_path),
        "config_digest": shared_resolved["config_digest"],
        "cell_eval2_version": shared_resolved["cell_eval2_version"],
        "pdex_version": shared_resolved["pdex_version"],
        "anchor_semantic_identity": run_identities[EXPECTED_LABELS[0]][
            "anchor_semantic_identity"
        ],
        "source_fingerprint": run_identities[EXPECTED_LABELS[0]]["source_fingerprint"],
        "full_generation_summary": full_summary,
        "generation_qc": generation_qc,
        "predictions": predictions,
        "outputs": validated_outputs,
    }


def validate_receipts(entries: list[tuple[Path, str]]) -> dict[str, Any]:
    require(bool(entries), "At least one scoring receipt is required")
    contexts = [context for _, context in entries]
    require(len(contexts) == len(set(contexts)), "Duplicate receipt context")
    require(
        set(contexts) == set(EXPECTED_CONTEXTS),
        "Receipt contexts must be exactly HepG2 and Jurkat",
    )
    validated = {
        context: validate_scoring_receipt(path, context) for path, context in entries
    }
    shared_fields = (
        "config_digest",
        "cell_eval2_version",
        "pdex_version",
        "anchor_semantic_identity",
    )
    for field in shared_fields:
        require(
            len({record[field] for record in validated.values()}) == 1,
            f"Cross-context scoring identity differs: {field}",
        )
    summary_identities = {
        (
            record["full_generation_summary"]["path"],
            record["full_generation_summary"]["sha256"],
        )
        for record in validated.values()
    }
    require(
        len(summary_identities) == 1,
        "Cross-context full-generation summary identities differ",
    )
    summary_path, summary_sha256 = next(iter(summary_identities))
    return {
        "schema": REPORT_SCHEMA,
        "contexts": validated,
        "shared_full_generation_summary": {
            "path": summary_path,
            "sha256": summary_sha256,
        },
        "shared_scoring_identity": {
            field: next(iter(validated.values()))[field] for field in shared_fields
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--receipt",
        action="append",
        nargs=2,
        metavar=("PATH", "CONTEXT"),
        required=True,
        help="Repeat for the HepG2 and Jurkat completed scoring receipts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    entries = [(Path(path), context) for path, context in args.receipt]
    print(json.dumps(validate_receipts(entries), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
