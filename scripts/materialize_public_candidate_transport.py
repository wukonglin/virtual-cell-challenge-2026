#!/usr/bin/env python3
"""Authenticate and materialize a truth-blind Anvil candidate on local storage.

The only external trust anchor accepted by ``materialize`` is the SHA-256 of
Anvil's completed ``anvil_candidate_completion.json``.  Its canonical binding
authenticates the inference runtime, checkpoint selection, candidate spec,
generation report, prediction, and generation verification.  The spec in turn
binds the two V5.1 transport sidecars and every canonical local generator input.

The completion receipt, inference runtime, checkpoint selection, prediction,
and sidecars are hard links: their bytes are never decoded, transformed, or
copied.  JSON changes are confined to an explicit path allowlist, regenerated
receipt hashes, and ``transport_provenance``.  No treated-truth path is
accepted by this command-line interface or constructed by this module.

The staging directory passed as ``--source-run-dir`` must contain these exact
relative paths (pull them with ``rsync -a``):

* ``anvil_candidate_completion.json``
* ``anvil_inference_runtime.json``
* ``spec.json``, ``generation.json``, ``prediction.h5ad``, and
  ``generation_verified.json``
* ``transport/split_manifest.json`` and
  ``transport/hepg2_state_residual_top100_v1.json``
* the completion-bound external selection receipt staged as
  ``transport/selected_checkpoint.json``
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Any, Sequence


SPEC_SCHEMA = "vcc-public-candidate-spec-v2"
GENERATION_SCHEMA = "vcc-native-state-anchor-counts-v1"
VERIFICATION_SCHEMA = "vcc-public-candidate-verification-v1"
ANVIL_COMPLETION_SCHEMA = "vcc-anvil-candidate-completion-v1"
ANVIL_FIREWALL_SCHEMA = "vcc-anvil-candidate-truth-firewall-v1"
INFERENCE_RUNTIME_SCHEMA = "vcc-anvil-state-reconstructed-inference-runtime-v2"
SELECTION_SCHEMA = "vcc-state-checkpoint-selection-v1"
PANEL_SCHEMA = "vcc-public-validation-manifest-v1"
RESIDUAL_SCHEMA = "vcc-public-state-residual-v1"
V51_TRANSPORT_SCHEMA = "vcc-public-v51-transport-rebase-v1"
DOCUMENT_TRANSPORT_SCHEMA = "vcc-public-candidate-document-rebase-v1"
TRANSPORT_RECEIPT_SCHEMA = "vcc-public-candidate-transport-receipt-v1"
TRANSPORT_RECEIPT_NAME = "candidate_transport_verified.json"
TRANSPORT_LINEAGE = "anvil-to-local-truth-blind-reconstruction-v1"
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
TAG_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,63}\Z")
CONTEXT_PREFIX = {"HepG2": "hepg2", "Jurkat": "jurkat"}
MAX_JSON_BYTES = 16 << 20
EXPECTED_VALIDATION_STEPS = tuple(range(2_000, 20_001, 2_000))
COMPLETION_ARTIFACT_MAP = {
    "inference_runtime": "inference_runtime",
    "checkpoint_selection": "checkpoint_selection",
    "candidate_spec": "spec",
    "generation_report": "generation_json",
    "prediction_h5ad": "prediction_h5ad",
    "generation_verification": "generation_verification",
}

CANONICAL_SPEC_INPUTS = (
    "controls_h5ad",
    "panel_csv",
    "residual_npz",
    "support_genes",
)
SIDECAR_SPEC_INPUTS = ("panel_manifest", "residual_json")
CARRIED_SPEC_INPUTS = ("selection_json",)
SPEC_PATH_INPUTS = (
    *CANONICAL_SPEC_INPUTS,
    *SIDECAR_SPEC_INPUTS,
    *CARRIED_SPEC_INPUTS,
)
GENERATION_PROVENANCE_MAP = {
    "controls_h5ad": "controls_h5ad",
    "panel_manifest": "panel_manifest",
    "panel_csv": "panel_csv",
    "support_gene_axis": "support_genes",
    "residual_artifact": "residual_npz",
    "checkpoint_selection": "selection_json",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if SHA256_PATTERN.fullmatch(normalized) is None:
        raise argparse.ArgumentTypeError(
            "expected exactly 64 hexadecimal SHA-256 characters"
        )
    return normalized


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
    require(
        stat.S_ISREG(resolved.stat().st_mode),
        f"{label} is not a regular file: {resolved}",
    )
    return resolved


def regular_directory(path: Path, label: str) -> Path:
    require(not path.is_symlink(), f"{label} must not be a symbolic link: {path}")
    try:
        resolved = path.resolve(strict=True)
    except FileNotFoundError as error:
        raise RuntimeError(f"Missing {label}: {path}") from error
    require(resolved.is_dir(), f"{label} is not a directory: {resolved}")
    return resolved


def describe_file(path: Path, *, recorded_path: Path | None = None) -> dict[str, Any]:
    resolved = regular_file(path, "provenance file")
    return {
        "path": str((recorded_path or resolved).absolute()),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def descriptor_at_path(source: Path, recorded_path: Path) -> dict[str, Any]:
    return describe_file(source, recorded_path=recorded_path)


def load_json(path: Path, label: str) -> dict[str, Any]:
    resolved = regular_file(path, label)
    require(resolved.stat().st_size <= MAX_JSON_BYTES, f"{label} is unexpectedly large")
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Unable to read {label}: {resolved}") from error
    require(type(value) is dict, f"{label} root must be a JSON object")
    return value


def descriptor_content(descriptor: Any, label: str) -> tuple[int, str]:
    require(type(descriptor) is dict, f"{label} descriptor is absent")
    path = descriptor.get("path")
    require(
        type(path) is str and Path(path).is_absolute(),
        f"{label} descriptor path must be absolute",
    )
    try:
        size = int(descriptor.get("size_bytes", -1))
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} descriptor size is invalid") from error
    digest = str(descriptor.get("sha256", "")).lower()
    require(size >= 0, f"{label} descriptor size is invalid")
    require(
        SHA256_PATTERN.fullmatch(digest) is not None,
        f"{label} descriptor SHA-256 is invalid",
    )
    return size, digest


def require_same_content(left: Any, right: Any, label: str) -> None:
    require(
        descriptor_content(left, f"{label} left")
        == descriptor_content(right, f"{label} right"),
        f"{label} content identity differs",
    )


def authenticate_file(path: Path, expected: Any, label: str) -> dict[str, Any]:
    observed = describe_file(regular_file(path, label))
    require_same_content(observed, expected, label)
    return observed


def require_descriptor_basename(descriptor: Any, expected: str, label: str) -> None:
    descriptor_content(descriptor, label)
    require(
        Path(str(descriptor["path"])).name == expected,
        f"{label} does not name {expected}",
    )


def canonical_binding_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_job_id(value: Any, label: str, *, allow_zero: bool = False) -> str:
    require(type(value) is str, f"{label} must be a string")
    pattern = r"0|[1-9][0-9]*" if allow_zero else r"[1-9][0-9]*"
    require(re.fullmatch(pattern, value) is not None, f"{label} is invalid")
    return value


def _validate_completion_descriptor(descriptor: Any, label: str) -> None:
    descriptor_content(descriptor, label)
    require(
        set(descriptor) == {"path", "size_bytes", "mtime_ns", "sha256"}
        and type(descriptor.get("size_bytes")) is int
        and descriptor["size_bytes"] > 0
        and type(descriptor.get("mtime_ns")) is int
        and descriptor["mtime_ns"] >= 0,
        f"{label} descriptor mtime_ns is invalid",
    )


def _validate_completion_payload(
    completion: dict[str, Any], *, context: str, output_tag: str
) -> dict[str, dict[str, Any]]:
    require(context == "HepG2", "Anvil completion transport supports only HepG2")
    expected_keys = {
        "schema",
        "output_tag",
        "context",
        "candidate_directory",
        "correlated_slurm_lineage",
        "truth_firewall",
        "artifacts",
        "completed_utc",
        "completion_state",
        "artifact_count",
        "producer",
        "binding",
    }
    require(
        set(completion) == expected_keys,
        "Anvil completion receipt fields are incomplete or unexpected",
    )
    require(
        completion.get("schema") == ANVIL_COMPLETION_SCHEMA,
        "Bad Anvil candidate completion schema",
    )
    require(
        completion.get("context") == context
        and completion.get("output_tag") == output_tag,
        "Anvil completion identity differs from the requested candidate",
    )
    candidate_directory = Path(str(completion.get("candidate_directory", "")))
    require(
        candidate_directory.is_absolute()
        and candidate_directory.name == output_tag
        and candidate_directory.parent.name == CONTEXT_PREFIX[context],
        "Anvil completion candidate directory is not canonical",
    )
    require(
        completion.get("completion_state")
        == "complete_after_strict_generation_verification"
        and completion.get("artifact_count") == len(COMPLETION_ARTIFACT_MAP),
        "Anvil completion state or artifact count is invalid",
    )
    require(
        type(completion.get("completed_utc")) is str
        and bool(completion["completed_utc"]),
        "Anvil completion timestamp is absent",
    )

    binding = completion.get("binding")
    require(
        type(binding) is dict
        and binding
        == {
            "algorithm": "sha256",
            "coverage": "every_top_level_field_except_binding",
            "canonical_json_sha256": binding.get("canonical_json_sha256"),
        },
        "Anvil completion binding declaration is malformed",
    )
    binding_digest = str(binding.get("canonical_json_sha256", "")).lower()
    require(
        SHA256_PATTERN.fullmatch(binding_digest) is not None,
        "Anvil completion canonical binding digest is invalid",
    )
    bound_document = {key: value for key, value in completion.items() if key != "binding"}
    require(
        canonical_binding_sha256(bound_document) == binding_digest,
        "Anvil completion canonical binding SHA-256 mismatch",
    )

    firewall = completion.get("truth_firewall")
    require(
        firewall
        == {
            "schema": ANVIL_FIREWALL_SCHEMA,
            "candidate_generation_truth_blind": True,
            "sealed_treated_profiles_read": False,
            "truth_inputs_read": [],
            "scoring_invoked_by_launcher": False,
            "leaderboard_submission_authorized": False,
        },
        "Anvil completion truth firewall is malformed",
    )
    lineage = completion.get("correlated_slurm_lineage")
    require(type(lineage) is dict, "Anvil completion Slurm lineage is absent")
    expected_lineage_keys = {
        "dependency",
        "dependency_type",
        "training_array_job_id",
        "training_array_task_id",
        "training_task_job_id",
        "inference_array_job_id",
        "inference_array_task_id",
        "inference_task_job_id",
        "corresponding_array_task_authenticated",
    }
    require(set(lineage) == expected_lineage_keys, "Anvil completion Slurm lineage is malformed")
    training_array = _canonical_job_id(
        lineage.get("training_array_job_id"), "training array job ID"
    )
    training_task = _canonical_job_id(
        lineage.get("training_task_job_id"), "training task job ID"
    )
    inference_array = _canonical_job_id(
        lineage.get("inference_array_job_id"), "inference array job ID"
    )
    inference_task_job = _canonical_job_id(
        lineage.get("inference_task_job_id"), "inference task job ID"
    )
    training_index = _canonical_job_id(
        lineage.get("training_array_task_id"), "training array task ID", allow_zero=True
    )
    inference_index = _canonical_job_id(
        lineage.get("inference_array_task_id"), "inference array task ID", allow_zero=True
    )
    require(
        lineage.get("dependency") == f"aftercorr:{training_array}"
        and lineage.get("dependency_type") == "aftercorr"
        and training_index == inference_index
        and training_task != inference_task_job
        and training_array != inference_array
        and lineage.get("corresponding_array_task_authenticated") is True,
        "Anvil completion does not bind a correlated training/inference task",
    )

    artifacts = completion.get("artifacts")
    require(
        type(artifacts) is dict and set(artifacts) == set(COMPLETION_ARTIFACT_MAP),
        "Anvil completion artifact set is not the canonical six-artifact set",
    )
    expected_filenames = {
        "inference_runtime": "anvil_inference_runtime.json",
        "checkpoint_selection": "selected_checkpoint.json",
        "candidate_spec": "spec.json",
        "generation_report": "generation.json",
        "prediction_h5ad": "prediction.h5ad",
        "generation_verification": "generation_verified.json",
    }
    for key, descriptor in artifacts.items():
        _validate_completion_descriptor(descriptor, f"completion artifact {key}")
        recorded_path = Path(str(descriptor["path"]))
        require(
            recorded_path.name == expected_filenames[key],
            f"Completion artifact {key} has a noncanonical filename",
        )
        if key != "checkpoint_selection":
            require(
                recorded_path.parent == candidate_directory,
                f"Completion artifact {key} is outside its candidate directory",
            )
    _validate_completion_descriptor(completion.get("producer"), "completion producer")
    return artifacts


def authenticate_completion_artifact(
    path: Path, expected: dict[str, Any], label: str
) -> dict[str, Any]:
    observed = authenticate_file(path, expected, label)
    require(
        regular_file(path, label).stat().st_mtime_ns == expected.get("mtime_ns"),
        f"{label} mtime differs from the completion binding; pull with rsync -a",
    )
    return observed


def _canonical_inputs(project_dir: Path, context: str) -> dict[str, Path]:
    prefix = CONTEXT_PREFIX[context]
    return {
        "controls_h5ad": project_dir
        / "dataset"
        / "public_v51"
        / f"{prefix}_controls_only.h5ad",
        "panel_csv": project_dir / "dataset" / "public_v51" / "split_manifest.csv",
        "residual_npz": project_dir
        / "artifacts"
        / "public_v51"
        / f"{prefix}_state_residual_top100_v1.npz",
        "support_genes": project_dir
        / "dataset"
        / "state_support"
        / "extracted"
        / "gene_names.csv",
    }


def _source_sidecars(source_run: Path, context: str) -> dict[str, Path]:
    prefix = CONTEXT_PREFIX[context]
    return {
        "panel_manifest": source_run / "transport" / "split_manifest.json",
        "residual_json": source_run
        / "transport"
        / f"{prefix}_state_residual_top100_v1.json",
    }


def _destination_sidecars(destination_run: Path, context: str) -> dict[str, Path]:
    return _source_sidecars(destination_run, context)


def _carried_candidate_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "candidate_completion": run_dir / "anvil_candidate_completion.json",
        "inference_runtime": run_dir / "anvil_inference_runtime.json",
        "checkpoint_selection": run_dir / "transport" / "selected_checkpoint.json",
    }


def _validate_v51_sidecar(
    payload: dict[str, Any],
    *,
    expected_schema: str,
    provenance_key: str,
    bound_payload: dict[str, Any],
    context: str,
) -> None:
    require(payload.get("schema") == expected_schema, "Bad V5.1 sidecar schema")
    if expected_schema == PANEL_SCHEMA:
        require(
            (payload.get("selection_contract") or {}).get("effect_values_accessed")
            is False,
            "Panel sidecar is not truth-blind",
        )
    else:
        require(payload.get("context") == context, "Residual sidecar context differs")
        require(
            (payload.get("contract") or {}).get("recipient_treated_profiles_used")
            is False,
            "Residual sidecar declares recipient treated profiles",
        )
    recorded = (payload.get("provenance") or {}).get(provenance_key)
    require_same_content(recorded, bound_payload, "V5.1 sidecar payload")
    transport = payload.get("transport_provenance")
    require(type(transport) is dict, "V5.1 sidecar lacks transport provenance")
    require(
        transport.get("schema") == V51_TRANSPORT_SCHEMA,
        "Bad V5.1 sidecar transport schema",
    )
    require(
        transport.get("payload_bytes_preserved") is True
        and transport.get("scientific_content_changed") is False,
        "V5.1 sidecar does not attest a byte-preserving path-only rebase",
    )


def _authenticate_source_bundle(
    source_run: Path,
    *,
    expected_completion_sha256: str,
    context: str,
    output_tag: str,
) -> dict[str, Any]:
    source_run = regular_directory(source_run, "source candidate directory")
    require(source_run.name == output_tag, "Source directory name differs from output tag")
    core_paths = {
        "candidate_completion": regular_file(
            source_run / "anvil_candidate_completion.json",
            "source Anvil completion receipt",
        ),
        "inference_runtime": regular_file(
            source_run / "anvil_inference_runtime.json", "source inference runtime"
        ),
        "checkpoint_selection": regular_file(
            source_run / "transport" / "selected_checkpoint.json",
            "source checkpoint selection",
        ),
        "spec": regular_file(source_run / "spec.json", "source spec"),
        "generation_json": regular_file(
            source_run / "generation.json", "source generation report"
        ),
        "prediction_h5ad": regular_file(
            source_run / "prediction.h5ad", "source prediction"
        ),
        "generation_verification": regular_file(
            source_run / "generation_verified.json", "source verification receipt"
        ),
    }
    observed_anchor = describe_file(core_paths["candidate_completion"])
    require(
        observed_anchor["sha256"] == expected_completion_sha256,
        "Source completion SHA-256 differs from the independent trust anchor",
    )
    completion = load_json(
        core_paths["candidate_completion"], "source Anvil completion receipt"
    )
    completion_artifacts = _validate_completion_payload(
        completion, context=context, output_tag=output_tag
    )
    authenticated_descriptors = {"candidate_completion": observed_anchor}
    for completion_key, local_key in COMPLETION_ARTIFACT_MAP.items():
        authenticated_descriptors[local_key] = authenticate_completion_artifact(
            core_paths[local_key],
            completion_artifacts[completion_key],
            f"source completion artifact {completion_key}",
        )

    runtime = load_json(core_paths["inference_runtime"], "source inference runtime")
    selection = load_json(
        core_paths["checkpoint_selection"], "source checkpoint selection"
    )
    verification = load_json(
        core_paths["generation_verification"], "source verification receipt"
    )
    require(
        verification.get("schema") == VERIFICATION_SCHEMA,
        "Bad source verification schema",
    )
    spec = load_json(core_paths["spec"], "source spec")
    generation = load_json(core_paths["generation_json"], "source generation report")
    require("transport_provenance" not in spec, "Refusing to retransport a candidate spec")
    require(
        "transport_provenance" not in generation,
        "Refusing to retransport a generation report",
    )
    require(
        "transport_provenance" not in verification,
        "Refusing to retransport a verification receipt",
    )
    require(
        runtime.get("schema") == INFERENCE_RUNTIME_SCHEMA
        and runtime.get("output_tag") == output_tag
        and runtime.get("leaderboard_submission_authorized") is False,
        "Source inference runtime identity or authorization is invalid",
    )
    require(selection.get("schema") == SELECTION_SCHEMA, "Bad source selection schema")
    require(
        selection.get("checkpoint_layout") == "validation_step_archives"
        and selection.get("unarchived_validation_steps") == [],
        "Source selection does not bind every validation archive",
    )
    from anvil_inference_lineage import (  # noqa: PLC0415
        validate_selected_earliest_argmin,
    )

    try:
        selected = validate_selected_earliest_argmin(
            selection.get("validation_results"),
            selection.get("archived_candidates"),
            selection.get("selected"),
            expected_steps=EXPECTED_VALIDATION_STEPS,
        )
    except RuntimeError as error:
        raise RuntimeError(f"Source checkpoint selection is invalid: {error}") from error
    require(spec.get("schema") == SPEC_SCHEMA, "Bad source candidate spec schema")
    require(generation.get("schema") == GENERATION_SCHEMA, "Bad source generation schema")
    require(
        spec.get("context") == generation.get("configuration", {}).get("context")
        == verification.get("context")
        == context,
        "Source candidate context differs from the expected context",
    )
    require(
        spec.get("output_tag") == verification.get("output_tag") == output_tag,
        "Source candidate output tag differs from the expected tag",
    )
    require(
        verification.get("status") == "passed"
        and verification.get("validation_mode") == "strict-v2",
        "Source candidate lacks a passed strict-v2 verification",
    )
    checks = verification.get("checks")
    require(
        type(checks) is dict
        and bool(checks)
        and all(value is True for value in checks.values())
        and checks.get("strict_generator_provenance_authenticated") is True,
        "Source candidate verification checks are incomplete",
    )
    require(
        verification.get("configuration") == spec.get("configuration"),
        "Source verification configuration differs from the spec",
    )
    require(
        (generation.get("data_firewall") or {}).get(
            "sealed_treated_profiles_read"
        )
        is False
        and (generation.get("data_firewall") or {}).get("truth_inputs_read") == [],
        "Source generation report does not satisfy the truth firewall",
    )
    require(
        (spec.get("firewall") or {}).get("generator_inputs_include_sealed_truth")
        is False
        and (spec.get("firewall") or {}).get("treated_profiles_read_while_planning")
        is False,
        "Source spec does not satisfy the truth firewall",
    )

    # Reuse the production structural contract.  This authenticates every
    # report/spec relationship without opening any path named inside them.
    from public_candidate_v6_contract import (  # noqa: PLC0415
        _validate_strict_spec_provenance,
        validate_generation_report,
    )

    _validate_strict_spec_provenance(spec)
    validate_generation_report(generation, spec)

    receipt_provenance = verification.get("provenance") or {}
    for key, filename, completion_key in (
        ("spec", "spec.json", "candidate_spec"),
        ("generation_json", "generation.json", "generation_report"),
        ("prediction_h5ad", "prediction.h5ad", "prediction_h5ad"),
    ):
        require_descriptor_basename(receipt_provenance.get(key), filename, key)
        require_same_content(
            receipt_provenance.get(key),
            completion_artifacts[completion_key],
            f"verification/completion {key}",
        )
    require_same_content(
        (generation.get("provenance") or {}).get("output_h5ad"),
        completion_artifacts["prediction_h5ad"],
        "generation/completion prediction",
    )

    selection_descriptor = completion_artifacts["checkpoint_selection"]
    require_same_content(
        (spec.get("inputs") or {}).get("selection_json"),
        selection_descriptor,
        "spec/completion selection",
    )
    require_same_content(
        (generation.get("provenance") or {}).get("checkpoint_selection"),
        selection_descriptor,
        "generation/completion selection",
    )
    runtime_selection = runtime.get("checkpoint_selection") or {}
    require_same_content(
        runtime_selection.get("selection_json"),
        selection_descriptor,
        "runtime/completion selection",
    )
    require_same_content(
        runtime_selection.get("checkpoint"),
        (spec.get("inputs") or {}).get("checkpoint"),
        "runtime/spec checkpoint",
    )
    require_same_content(
        runtime_selection.get("checkpoint"),
        (generation.get("provenance") or {}).get("checkpoint"),
        "runtime/generation checkpoint",
    )
    require(
        runtime_selection.get("expected_checkpoint_sha256")
        == runtime_selection["checkpoint"]["sha256"],
        "Runtime expected checkpoint SHA-256 differs from its descriptor",
    )
    runtime_transported = runtime.get("transported_inputs") or {}
    require_same_content(
        runtime_transported.get("panel_json"),
        (spec.get("inputs") or {}).get("panel_manifest"),
        "runtime/spec panel sidecar",
    )
    require_same_content(
        runtime_transported.get("residual_json"),
        (spec.get("inputs") or {}).get("residual_json"),
        "runtime/spec residual sidecar",
    )
    selected_step = selected["global_step"]
    selected_loss = selected["val_loss"]
    require(
        (spec.get("checkpoint_selection") or {}).get("global_step") == selected_step
        and (spec.get("checkpoint_selection") or {}).get("val_loss") == selected_loss
        and (generation.get("checkpoint_selection") or {}).get("global_step")
        == selected_step
        and (generation.get("checkpoint_selection") or {}).get("val_loss")
        == selected_loss
        and runtime_selection.get("selected_step") == selected_step
        and runtime_selection.get("selected_val_loss") == selected_loss,
        "Selected checkpoint differs across completion-bound artifacts",
    )
    completion_lineage = completion["correlated_slurm_lineage"]
    runtime_slurm = runtime.get("slurm") or {}
    runtime_upstream = runtime_slurm.get("correlated_upstream") or {}
    require(
        runtime_slurm.get("job_id") == completion_lineage["inference_task_job_id"]
        and runtime_slurm.get("array_job_id")
        == completion_lineage["inference_array_job_id"]
        and runtime_slurm.get("array_task_id")
        == completion_lineage["inference_array_task_id"]
        and runtime_slurm.get("job_dependency") == completion_lineage["dependency"]
        and runtime_slurm.get("declared_training_array_job_id")
        == completion_lineage["training_array_job_id"]
        and runtime_upstream.get("array_job_id")
        == completion_lineage["training_array_job_id"]
        and runtime_upstream.get("array_task_id")
        == completion_lineage["training_array_task_id"]
        and runtime_upstream.get("task_job_id")
        == completion_lineage["training_task_job_id"]
        and runtime_upstream.get("dependency_type") == "aftercorr",
        "Inference runtime Slurm lineage differs from the completion binding",
    )

    inputs = spec.get("inputs") or {}
    sidecar_paths = _source_sidecars(source_run, context)
    sidecars: dict[str, dict[str, Any]] = {}
    for key, path in sidecar_paths.items():
        resolved = regular_file(path, f"source {key}")
        authenticated_descriptors[key] = authenticate_file(
            resolved, inputs.get(key), f"source {key}"
        )
        sidecars[key] = load_json(resolved, f"source {key}")
    _validate_v51_sidecar(
        sidecars["panel_manifest"],
        expected_schema=PANEL_SCHEMA,
        provenance_key="output_csv",
        bound_payload=inputs.get("panel_csv"),
        context=context,
    )
    _validate_v51_sidecar(
        sidecars["residual_json"],
        expected_schema=RESIDUAL_SCHEMA,
        provenance_key="output_npz",
        bound_payload=inputs.get("residual_npz"),
        context=context,
    )
    return {
        "source_run": source_run,
        "paths": {**core_paths, **sidecar_paths},
        "completion": completion,
        "runtime": runtime,
        "selection": selection,
        "spec": spec,
        "generation": generation,
        "verification": verification,
        "sidecars": sidecars,
        "descriptors": authenticated_descriptors,
        "expected_completion_sha256": expected_completion_sha256,
    }


def _authenticate_canonical_inputs(
    project_dir: Path, context: str, source_spec: dict[str, Any]
) -> tuple[dict[str, Path], dict[str, dict[str, Any]]]:
    paths = _canonical_inputs(project_dir, context)
    descriptors: dict[str, dict[str, Any]] = {}
    source_inputs = source_spec.get("inputs") or {}
    for key, path in paths.items():
        resolved = regular_file(path, f"canonical local {key}")
        descriptors[key] = authenticate_file(
            resolved, source_inputs.get(key), f"canonical local {key}"
        )
        paths[key] = resolved
    return paths, descriptors


def _rebase_descriptor(
    source_descriptor: dict[str, Any], destination_path: Path
) -> dict[str, Any]:
    descriptor_content(source_descriptor, "source rebase")
    rebased = copy.deepcopy(source_descriptor)
    rebased["path"] = str(destination_path.absolute())
    return rebased


def _path_change(
    *, document: str, pointer: str, source: dict[str, Any], destination: dict[str, Any]
) -> dict[str, Any]:
    require_same_content(source, destination, f"{document} {pointer}")
    return {
        "document": document,
        "json_pointer": pointer,
        "source_path": source["path"],
        "destination_path": destination["path"],
        "size_bytes": int(destination["size_bytes"]),
        "sha256": destination["sha256"],
    }


def _document_transport(
    *, source_document: dict[str, Any], trust_anchor: dict[str, Any], pointers: list[str]
) -> dict[str, Any]:
    return {
        "schema": DOCUMENT_TRANSPORT_SCHEMA,
        "lineage": TRANSPORT_LINEAGE,
        "operation": "authenticated-path-only-rebase",
        "source_document": source_document,
        "source_completion_trust_anchor": trust_anchor,
        "rebased_json_pointers": sorted(pointers),
        "prediction_bytes_preserved": True,
        "scientific_content_changed": False,
        "treated_truth_read": False,
        "treated_truth_copied": False,
        "treated_truth_exported": False,
    }


def _build_rebased_spec(
    source: dict[str, Any],
    destinations: dict[str, dict[str, Any]],
    *,
    source_descriptor: dict[str, Any],
    trust_anchor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rebased = copy.deepcopy(source)
    changes: list[dict[str, Any]] = []
    pointers: list[str] = []
    for key in SPEC_PATH_INPUTS:
        before = source["inputs"][key]
        after = destinations[key]
        rebased["inputs"][key] = after
        pointer = f"/inputs/{key}/path"
        pointers.append(pointer)
        changes.append(
            _path_change(document="spec.json", pointer=pointer, source=before, destination=after)
        )
    rebased["transport_provenance"] = _document_transport(
        source_document=source_descriptor,
        trust_anchor=trust_anchor,
        pointers=pointers,
    )
    return rebased, changes


def _build_rebased_generation(
    source: dict[str, Any],
    spec_destinations: dict[str, dict[str, Any]],
    prediction_destination: dict[str, Any],
    *,
    source_descriptor: dict[str, Any],
    trust_anchor: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rebased = copy.deepcopy(source)
    changes: list[dict[str, Any]] = []
    pointers: list[str] = []
    provenance = rebased["provenance"]
    source_provenance = source["provenance"]
    for report_key, spec_key in GENERATION_PROVENANCE_MAP.items():
        before = source_provenance.get(report_key)
        if before is None:
            require(
                report_key == "residual_artifact"
                and float(source["configuration"]["residual_alpha"]) == 0.0,
                f"Source generation provenance unexpectedly lacks {report_key}",
            )
            continue
        after = spec_destinations[spec_key]
        provenance[report_key] = after
        pointer = f"/provenance/{report_key}/path"
        pointers.append(pointer)
        changes.append(
            _path_change(
                document="generation.json",
                pointer=pointer,
                source=before,
                destination=after,
            )
        )
    before_output = source_provenance["output_h5ad"]
    provenance["output_h5ad"] = prediction_destination
    output_pointer = "/provenance/output_h5ad/path"
    pointers.append(output_pointer)
    changes.append(
        _path_change(
            document="generation.json",
            pointer=output_pointer,
            source=before_output,
            destination=prediction_destination,
        )
    )
    residual = rebased.get("residual") or {}
    if float(source["configuration"]["residual_alpha"]) == 0.0:
        before_path = (source.get("residual") or {}).get("declared_path")
        after_path = spec_destinations["residual_npz"]["path"]
        require(
            type(before_path) is str and Path(before_path).is_absolute(),
            "Bad residual declaration",
        )
        residual["declared_path"] = after_path
        rebased["residual"] = residual
        pointer = "/residual/declared_path"
        pointers.append(pointer)
        changes.append(
            {
                "document": "generation.json",
                "json_pointer": pointer,
                "source_path": before_path,
                "destination_path": after_path,
                "size_bytes": spec_destinations["residual_npz"]["size_bytes"],
                "sha256": spec_destinations["residual_npz"]["sha256"],
            }
        )
    rebased["transport_provenance"] = _document_transport(
        source_document=source_descriptor,
        trust_anchor=trust_anchor,
        pointers=pointers,
    )
    return rebased, changes


def _build_rebased_verification(
    source: dict[str, Any],
    *,
    spec_destination: dict[str, Any],
    generation_destination: dict[str, Any],
    prediction_destination: dict[str, Any],
    source_descriptor: dict[str, Any],
    trust_anchor: dict[str, Any],
) -> dict[str, Any]:
    rebased = copy.deepcopy(source)
    rebased["provenance"]["spec"] = spec_destination
    rebased["provenance"]["generation_json"] = generation_destination
    rebased["provenance"]["prediction_h5ad"] = prediction_destination
    rebased["checks"]["authenticated_candidate_transport"] = True
    pointers = [
        "/provenance/spec",
        "/provenance/generation_json",
        "/provenance/prediction_h5ad/path",
        "/checks/authenticated_candidate_transport",
    ]
    rebased["transport_provenance"] = _document_transport(
        source_document=source_descriptor,
        trust_anchor=trust_anchor,
        pointers=pointers,
    )
    return rebased


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_new_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as handle:
        handle.write(_json_bytes(payload))
        handle.flush()
        os.fsync(handle.fileno())


def _path_descriptor_from_file(file_path: Path, final_path: Path) -> dict[str, Any]:
    return descriptor_at_path(file_path, final_path)


def _build_receipt(
    *,
    context: str,
    output_tag: str,
    source_run: Path,
    destination_run: Path,
    expected_completion_sha256: str,
    canonical_completion_binding_sha256: str,
    source_descriptors: dict[str, dict[str, Any]],
    destination_descriptors: dict[str, dict[str, Any]],
    canonical_descriptors: dict[str, dict[str, Any]],
    path_changes: list[dict[str, Any]],
) -> dict[str, Any]:
    completion_bound_artifacts = {
        completion_key: {
            "source": source_descriptors[local_key],
            "destination": destination_descriptors[local_key],
        }
        for completion_key, local_key in COMPLETION_ARTIFACT_MAP.items()
    }
    json_change_allowlist = {
        "spec.json": ["/transport_provenance"],
        "generation.json": ["/transport_provenance"],
        "generation_verified.json": [
            "/checks/authenticated_candidate_transport",
            "/provenance/generation_json",
            "/provenance/prediction_h5ad/path",
            "/provenance/spec",
            "/transport_provenance",
        ],
    }
    for change in path_changes:
        json_change_allowlist[change["document"]].append(change["json_pointer"])
    json_change_allowlist = {
        document: sorted(set(pointers))
        for document, pointers in json_change_allowlist.items()
    }
    return {
        "schema": TRANSPORT_RECEIPT_SCHEMA,
        "status": "passed",
        "lineage": TRANSPORT_LINEAGE,
        "operation": "authenticated-path-only-anvil-to-local-rebase",
        "context": context,
        "output_tag": output_tag,
        "source": {
            "run_directory": str(source_run),
            "independently_supplied_completion_sha256": expected_completion_sha256,
            "canonical_completion_binding_sha256": canonical_completion_binding_sha256,
            "artifacts": source_descriptors,
        },
        "destination": {
            "run_directory": str(destination_run),
            "artifacts": destination_descriptors,
        },
        "canonical_local_inputs": canonical_descriptors,
        "completion_bound_artifacts": completion_bound_artifacts,
        "path_rebases": sorted(
            path_changes, key=lambda item: (item["document"], item["json_pointer"])
        ),
        "json_change_allowlist": json_change_allowlist,
        "regenerated_authentication_fields": [
            "/generation_verified.json/provenance/spec",
            "/generation_verified.json/provenance/generation_json",
        ],
        "contract": {
            "source_completion_is_external_trust_anchor": True,
            "source_completion_canonical_binding_authenticated": True,
            "completion_bound_six_artifacts_authenticated": True,
            "source_bundle_authenticated_before_materialization": True,
            "source_files_reauthenticated_after_staging": True,
            "completion_is_hard_link_to_authenticated_source": True,
            "inference_runtime_is_hard_link_to_authenticated_source": True,
            "checkpoint_selection_is_hard_link_to_authenticated_source": True,
            "prediction_is_hard_link_to_authenticated_source": True,
            "prediction_bytes_preserved": True,
            "transport_sidecar_bytes_preserved": True,
            "json_changes_confined_to_recorded_allowlist": True,
            "scientific_content_changed": False,
            "treated_truth_read": False,
            "treated_truth_copied": False,
            "treated_truth_exported": False,
        },
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def materialize(
    *,
    project_dir: Path,
    source_run: Path,
    source_completion_expected_sha256: str,
    destination_public_root: Path,
    context: str,
    output_tag: str,
) -> Path:
    require(context in CONTEXT_PREFIX, "Unsupported context")
    require(TAG_PATTERN.fullmatch(output_tag) is not None, "Unsafe output tag")
    project_dir = regular_directory(project_dir, "project directory")
    source = _authenticate_source_bundle(
        source_run,
        expected_completion_sha256=source_completion_expected_sha256,
        context=context,
        output_tag=output_tag,
    )
    canonical_paths, canonical_descriptors = _authenticate_canonical_inputs(
        project_dir, context, source["spec"]
    )
    destination_public_root = destination_public_root.resolve(strict=False)
    artifact_root = (project_dir / "artifacts").resolve(strict=True)
    require(
        _is_relative_to(destination_public_root, artifact_root),
        "Destination public root must be inside the project artifacts directory",
    )
    destination_run = (
        destination_public_root
        / "candidates"
        / CONTEXT_PREFIX[context]
        / output_tag
    ).absolute()
    sensitive_root = (project_dir / "dataset" / "public_v51").resolve(strict=True)
    require(
        not _is_relative_to(source["source_run"], sensitive_root),
        "Source candidate must not be staged inside the public V5.1 data directory",
    )
    require(
        not _is_relative_to(destination_run, sensitive_root),
        "Destination must not overlap the public V5.1 data directory",
    )
    require(
        destination_run != source["source_run"]
        and not _is_relative_to(destination_run, source["source_run"])
        and not _is_relative_to(source["source_run"], destination_run),
        "Source and destination candidate directories must be disjoint",
    )
    require(
        not destination_run.exists() and not destination_run.is_symlink(),
        f"Refusing to overwrite destination candidate: {destination_run}",
    )
    destination_run.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_tag}.transport.", dir=destination_run.parent)
    )
    try:
        temporary_sidecars = _destination_sidecars(temporary, context)
        final_sidecars = _destination_sidecars(destination_run, context)
        temporary_carried = _carried_candidate_paths(temporary)
        final_carried = _carried_candidate_paths(destination_run)
        (temporary / "transport").mkdir()
        os.link(source["paths"]["prediction_h5ad"], temporary / "prediction.h5ad")
        for key in SIDECAR_SPEC_INPUTS:
            os.link(source["paths"][key], temporary_sidecars[key])
        for key in ("candidate_completion", "inference_runtime", "checkpoint_selection"):
            os.link(source["paths"][key], temporary_carried[key])

        spec_destinations = {
            key: _rebase_descriptor(source["spec"]["inputs"][key], canonical_paths[key])
            for key in CANONICAL_SPEC_INPUTS
        }
        for key in SIDECAR_SPEC_INPUTS:
            spec_destinations[key] = _rebase_descriptor(
                source["spec"]["inputs"][key], final_sidecars[key]
            )
        spec_destinations["selection_json"] = _rebase_descriptor(
            source["spec"]["inputs"]["selection_json"],
            final_carried["checkpoint_selection"],
        )
        prediction_destination = _path_descriptor_from_file(
            temporary / "prediction.h5ad", destination_run / "prediction.h5ad"
        )
        trust_anchor = source["descriptors"]["candidate_completion"]
        rebased_spec, spec_changes = _build_rebased_spec(
            source["spec"],
            spec_destinations,
            source_descriptor=source["descriptors"]["spec"],
            trust_anchor=trust_anchor,
        )
        rebased_generation, generation_changes = _build_rebased_generation(
            source["generation"],
            spec_destinations,
            prediction_destination,
            source_descriptor=source["descriptors"]["generation_json"],
            trust_anchor=trust_anchor,
        )
        _write_new_json(temporary / "spec.json", rebased_spec)
        _write_new_json(temporary / "generation.json", rebased_generation)
        spec_destination = _path_descriptor_from_file(
            temporary / "spec.json", destination_run / "spec.json"
        )
        generation_destination = _path_descriptor_from_file(
            temporary / "generation.json", destination_run / "generation.json"
        )
        rebased_verification = _build_rebased_verification(
            source["verification"],
            spec_destination=spec_destination,
            generation_destination=generation_destination,
            prediction_destination=prediction_destination,
            source_descriptor=source["descriptors"]["generation_verification"],
            trust_anchor=trust_anchor,
        )
        _write_new_json(temporary / "generation_verified.json", rebased_verification)

        destination_descriptors = {
            "spec": spec_destination,
            "generation_json": generation_destination,
            "prediction_h5ad": prediction_destination,
            "generation_verification": _path_descriptor_from_file(
                temporary / "generation_verified.json",
                destination_run / "generation_verified.json",
            ),
            "panel_manifest": _path_descriptor_from_file(
                temporary_sidecars["panel_manifest"], final_sidecars["panel_manifest"]
            ),
            "residual_json": _path_descriptor_from_file(
                temporary_sidecars["residual_json"], final_sidecars["residual_json"]
            ),
            "candidate_completion": _path_descriptor_from_file(
                temporary_carried["candidate_completion"],
                final_carried["candidate_completion"],
            ),
            "inference_runtime": _path_descriptor_from_file(
                temporary_carried["inference_runtime"],
                final_carried["inference_runtime"],
            ),
            "checkpoint_selection": _path_descriptor_from_file(
                temporary_carried["checkpoint_selection"],
                final_carried["checkpoint_selection"],
            ),
        }
        # Rehash every source artifact after all destination content is staged.
        for key, path in source["paths"].items():
            authenticate_file(path, source["descriptors"][key], f"restaged source {key}")
        for key, destination in {
            "prediction_h5ad": temporary / "prediction.h5ad",
            "candidate_completion": temporary_carried["candidate_completion"],
            "inference_runtime": temporary_carried["inference_runtime"],
            "checkpoint_selection": temporary_carried["checkpoint_selection"],
            **temporary_sidecars,
        }.items():
            require(
                os.path.samestat(source["paths"][key].stat(), destination.stat()),
                f"Destination {key} is not a hard link to the authenticated source",
            )
        receipt = _build_receipt(
            context=context,
            output_tag=output_tag,
            source_run=source["source_run"],
            destination_run=destination_run,
            expected_completion_sha256=source_completion_expected_sha256,
            canonical_completion_binding_sha256=source["completion"]["binding"][
                "canonical_json_sha256"
            ],
            source_descriptors=source["descriptors"],
            destination_descriptors=destination_descriptors,
            canonical_descriptors=canonical_descriptors,
            path_changes=[*spec_changes, *generation_changes],
        )
        _write_new_json(temporary / TRANSPORT_RECEIPT_NAME, receipt)
        require(not destination_run.exists(), "Destination appeared during materialization")
        os.rename(temporary, destination_run)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    try:
        validate_transport_receipt(
            run_dir=destination_run,
            project_dir=project_dir,
            expected_context=context,
            expected_output_tag=output_tag,
        )
    except BaseException:
        # The directory was created by this invocation and has never been
        # returned to the caller; remove a failed post-publication audit.
        shutil.rmtree(destination_run)
        raise
    return destination_run / TRANSPORT_RECEIPT_NAME


def transport_is_required(
    *, run_dir: Path, spec: dict[str, Any], output_tag: str
) -> bool:
    state_source_path = str((spec.get("state_source") or {}).get("path", ""))
    return bool(
        (run_dir / TRANSPORT_RECEIPT_NAME).exists()
        or (run_dir / "anvil_candidate_completion.json").exists()
        or "transport_provenance" in spec
        or output_tag.startswith("state_sm_anvil_reconstructed_")
        or state_source_path == "/anvil"
        or state_source_path.startswith("/anvil/")
    )


def validate_transport_receipt(
    *,
    run_dir: Path,
    project_dir: Path,
    expected_context: str,
    expected_output_tag: str,
) -> dict[str, Any] | None:
    require(expected_context in CONTEXT_PREFIX, "Unsupported context")
    require(TAG_PATTERN.fullmatch(expected_output_tag) is not None, "Unsafe output tag")
    run_dir = regular_directory(run_dir, "candidate run directory")
    project_dir = regular_directory(project_dir, "project directory")
    spec = load_json(run_dir / "spec.json", "candidate spec")
    if not transport_is_required(run_dir=run_dir, spec=spec, output_tag=expected_output_tag):
        return None
    receipt_path = regular_file(
        run_dir / TRANSPORT_RECEIPT_NAME, "candidate transport receipt"
    )
    receipt = load_json(receipt_path, "candidate transport receipt")
    require(
        receipt.get("schema") == TRANSPORT_RECEIPT_SCHEMA
        and receipt.get("status") == "passed",
        "Candidate transport receipt is not a passed v1 receipt",
    )
    require(
        receipt.get("context") == expected_context
        and receipt.get("output_tag") == expected_output_tag,
        "Candidate transport receipt identity differs from the score request",
    )
    require(run_dir.name == expected_output_tag, "Candidate directory name differs from tag")
    require(
        Path(str((receipt.get("destination") or {}).get("run_directory", "")))
        == run_dir,
        "Transport receipt destination directory differs",
    )
    expected_anchor = str(
        (receipt.get("source") or {}).get(
            "independently_supplied_completion_sha256", ""
        )
    ).lower()
    require(
        SHA256_PATTERN.fullmatch(expected_anchor) is not None,
        "Transport receipt source trust anchor is invalid",
    )
    source_run = Path(str((receipt.get("source") or {}).get("run_directory", "")))
    source = _authenticate_source_bundle(
        source_run,
        expected_completion_sha256=expected_anchor,
        context=expected_context,
        output_tag=expected_output_tag,
    )
    canonical_paths, canonical_descriptors = _authenticate_canonical_inputs(
        project_dir, expected_context, source["spec"]
    )
    destination_paths = {
        "spec": regular_file(run_dir / "spec.json", "transported spec"),
        "generation_json": regular_file(
            run_dir / "generation.json", "transported generation report"
        ),
        "prediction_h5ad": regular_file(
            run_dir / "prediction.h5ad", "transported prediction"
        ),
        "generation_verification": regular_file(
            run_dir / "generation_verified.json", "transported verification"
        ),
        **_destination_sidecars(run_dir, expected_context),
        **_carried_candidate_paths(run_dir),
    }
    for key in (
        *SIDECAR_SPEC_INPUTS,
        "candidate_completion",
        "inference_runtime",
        "checkpoint_selection",
    ):
        destination_paths[key] = regular_file(
            destination_paths[key], f"transported {key}"
        )
    destination_descriptors = {
        key: describe_file(path) for key, path in destination_paths.items()
    }
    for key in (
        *SIDECAR_SPEC_INPUTS,
        "candidate_completion",
        "inference_runtime",
        "checkpoint_selection",
    ):
        require_same_content(
            destination_descriptors[key], source["descriptors"][key], key
        )
        require(
            os.path.samestat(destination_paths[key].stat(), source["paths"][key].stat()),
            f"Transported {key} is not a hard link to its authenticated source",
        )
    require_same_content(
        destination_descriptors["prediction_h5ad"],
        source["descriptors"]["prediction_h5ad"],
        "transported prediction",
    )
    require(
        os.path.samestat(
            destination_paths["prediction_h5ad"].stat(),
            source["paths"]["prediction_h5ad"].stat(),
        ),
        "Transported prediction is not a hard link to its authenticated source",
    )

    final_sidecars = _destination_sidecars(run_dir, expected_context)
    spec_destinations = {
        key: _rebase_descriptor(source["spec"]["inputs"][key], canonical_paths[key])
        for key in CANONICAL_SPEC_INPUTS
    }
    for key in SIDECAR_SPEC_INPUTS:
        spec_destinations[key] = _rebase_descriptor(
            source["spec"]["inputs"][key], final_sidecars[key]
        )
    final_carried = _carried_candidate_paths(run_dir)
    spec_destinations["selection_json"] = _rebase_descriptor(
        source["spec"]["inputs"]["selection_json"],
        final_carried["checkpoint_selection"],
    )
    trust_anchor = source["descriptors"]["candidate_completion"]
    expected_spec, spec_changes = _build_rebased_spec(
        source["spec"],
        spec_destinations,
        source_descriptor=source["descriptors"]["spec"],
        trust_anchor=trust_anchor,
    )
    require(spec == expected_spec, "Transported spec changed outside its allowlist")
    generation = load_json(destination_paths["generation_json"], "transported generation")
    expected_generation, generation_changes = _build_rebased_generation(
        source["generation"],
        spec_destinations,
        destination_descriptors["prediction_h5ad"],
        source_descriptor=source["descriptors"]["generation_json"],
        trust_anchor=trust_anchor,
    )
    require(
        generation == expected_generation,
        "Transported generation report changed outside its allowlist",
    )
    verification = load_json(
        destination_paths["generation_verification"], "transported verification"
    )
    expected_verification = _build_rebased_verification(
        source["verification"],
        spec_destination=destination_descriptors["spec"],
        generation_destination=destination_descriptors["generation_json"],
        prediction_destination=destination_descriptors["prediction_h5ad"],
        source_descriptor=source["descriptors"]["generation_verification"],
        trust_anchor=trust_anchor,
    )
    require(
        verification == expected_verification,
        "Transported verification changed outside its regeneration allowlist",
    )
    expected_receipt = _build_receipt(
        context=expected_context,
        output_tag=expected_output_tag,
        source_run=source["source_run"],
        destination_run=run_dir,
        expected_completion_sha256=expected_anchor,
        canonical_completion_binding_sha256=source["completion"]["binding"][
            "canonical_json_sha256"
        ],
        source_descriptors=source["descriptors"],
        destination_descriptors=destination_descriptors,
        canonical_descriptors=canonical_descriptors,
        path_changes=[*spec_changes, *generation_changes],
    )
    require(receipt == expected_receipt, "Candidate transport receipt content differs")
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser(
        "materialize", help="Authenticate one exported bundle and publish a local candidate."
    )
    create.add_argument("--project-dir", type=Path, required=True)
    create.add_argument("--source-run-dir", type=Path, required=True)
    create.add_argument(
        "--source-completion-expected-sha256", type=parse_sha256, required=True
    )
    create.add_argument("--destination-public-root", type=Path, required=True)
    create.add_argument("--expected-context", choices=("HepG2",), required=True)
    create.add_argument("--expected-output-tag", required=True)

    verify = subparsers.add_parser(
        "verify", help="Fail closed if a candidate requires an invalid transport receipt."
    )
    verify.add_argument("--project-dir", type=Path, required=True)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--expected-context", choices=tuple(CONTEXT_PREFIX), required=True)
    verify.add_argument("--expected-output-tag", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "materialize":
        receipt_path = materialize(
            project_dir=args.project_dir,
            source_run=args.source_run_dir,
            source_completion_expected_sha256=args.source_completion_expected_sha256,
            destination_public_root=args.destination_public_root,
            context=args.expected_context,
            output_tag=args.expected_output_tag,
        )
        payload = load_json(receipt_path, "candidate transport receipt")
    else:
        payload = validate_transport_receipt(
            run_dir=args.run_dir,
            project_dir=args.project_dir,
            expected_context=args.expected_context,
            expected_output_tag=args.expected_output_tag,
        )
        if payload is None:
            payload = {
                "schema": TRANSPORT_RECEIPT_SCHEMA,
                "status": "not-required",
                "context": args.expected_context,
                "output_tag": args.expected_output_tag,
            }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
