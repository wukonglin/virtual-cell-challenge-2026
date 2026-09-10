#!/usr/bin/env python3
"""Publish final authenticated completion evidence for an Anvil candidate."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from anvil_inference_lineage import validate_selected_earliest_argmin


SCHEMA = "vcc-anvil-candidate-completion-v1"
FIREWALL_SCHEMA = "vcc-anvil-candidate-truth-firewall-v1"
INFERENCE_RUNTIME_SCHEMA = "vcc-anvil-state-reconstructed-inference-runtime-v2"
SELECTION_SCHEMA = "vcc-state-checkpoint-selection-v1"
SPEC_SCHEMA = "vcc-public-candidate-spec-v2"
GENERATION_SCHEMA = "vcc-native-state-anchor-counts-v1"
VERIFICATION_SCHEMA = "vcc-public-candidate-verification-v1"
EXPECTED_CONTEXT = "HepG2"
EXPECTED_VALIDATION_STEPS = tuple(range(2_000, 20_001, 2_000))


class CompletionError(RuntimeError):
    """A final candidate-completion precondition failed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CompletionError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    require(isinstance(value, Mapping), f"{label} must be a mapping")
    return value


def _canonical_job_id(value: Any, label: str, *, allow_zero: bool = False) -> str:
    require(isinstance(value, str), f"{label} must be a string")
    pattern = r"0|[1-9][0-9]*" if allow_zero else r"[1-9][0-9]*"
    require(re.fullmatch(pattern, value) is not None, f"{label} is invalid")
    return value


def _regular_file_descriptor(path: Path, label: str) -> dict[str, Any]:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    before = path.lstat()
    require(
        stat.S_ISREG(before.st_mode) and not path.is_symlink() and before.st_size > 0,
        f"{label} must be a nonempty regular file",
    )
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    after = path.lstat()
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    require(before_identity == after_identity, f"{label} changed while it was hashed")
    return {
        "path": str(path.resolve(strict=True)),
        "size_bytes": int(after.st_size),
        "mtime_ns": int(after.st_mtime_ns),
        "sha256": digest.hexdigest(),
    }


def _load_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    descriptor = _regular_file_descriptor(path, label)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CompletionError(f"Cannot read {label} JSON: {error}") from error
    require(isinstance(payload, dict), f"{label} JSON root must be a mapping")
    # Detect any change between authenticated hashing and JSON decoding.
    require(
        _regular_file_descriptor(path, label) == descriptor,
        f"{label} changed after it was hashed",
    )
    return payload, descriptor


def _descriptor_matches(
    recorded: Any,
    authenticated: Mapping[str, Any],
    label: str,
) -> None:
    value = _mapping(recorded, f"recorded {label} descriptor")
    try:
        recorded_path = str(Path(str(value.get("path", ""))).resolve(strict=True))
    except (OSError, RuntimeError) as error:
        raise CompletionError(f"Recorded {label} path cannot be resolved") from error
    require(recorded_path == authenticated["path"], f"{label} path mismatch")
    require(value.get("size_bytes") == authenticated["size_bytes"], f"{label} size mismatch")
    require(value.get("sha256") == authenticated["sha256"], f"{label} hash mismatch")
    if "mtime_ns" in value:
        require(value.get("mtime_ns") == authenticated["mtime_ns"], f"{label} mtime mismatch")


def canonical_binding_sha256(binding: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        binding,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_completion_payload(
    *,
    runtime_path: Path,
    selection_path: Path,
    spec_path: Path,
    generation_path: Path,
    prediction_path: Path,
    verification_path: Path,
    output_tag: str,
    context: str,
    environment: Mapping[str, str],
    producer_path: Path,
) -> dict[str, Any]:
    """Authenticate all final artifacts and construct their completion receipt."""

    require(context == EXPECTED_CONTEXT, f"context must be {EXPECTED_CONTEXT}")
    require(re.fullmatch(r"[a-z0-9][a-z0-9._-]{0,63}", output_tag) is not None, "invalid output tag")

    runtime, runtime_descriptor = _load_json(runtime_path, "inference runtime")
    selection, selection_descriptor = _load_json(selection_path, "selection receipt")
    spec, spec_descriptor = _load_json(spec_path, "candidate spec")
    generation, generation_descriptor = _load_json(generation_path, "generation report")
    verification, verification_descriptor = _load_json(
        verification_path, "generation verification"
    )
    prediction_descriptor = _regular_file_descriptor(prediction_path, "prediction H5AD")
    producer_descriptor = _regular_file_descriptor(producer_path, "completion writer")

    candidate_dir = Path(runtime_descriptor["path"]).parent
    for label, descriptor in (
        ("candidate spec", spec_descriptor),
        ("generation report", generation_descriptor),
        ("prediction H5AD", prediction_descriptor),
        ("generation verification", verification_descriptor),
    ):
        require(
            Path(str(descriptor["path"])).parent == candidate_dir,
            f"{label} is outside the candidate directory",
        )

    require(runtime.get("schema") == INFERENCE_RUNTIME_SCHEMA, "bad inference runtime schema")
    require(runtime.get("output_tag") == output_tag, "runtime output tag mismatch")
    require(
        runtime.get("leaderboard_submission_authorized") is False,
        "runtime does not prohibit leaderboard submission",
    )
    runtime_slurm = _mapping(runtime.get("slurm"), "runtime Slurm lineage")
    inference_job_id = _canonical_job_id(environment.get("SLURM_JOB_ID"), "SLURM_JOB_ID")
    inference_array_job_id = _canonical_job_id(
        environment.get("SLURM_ARRAY_JOB_ID"), "SLURM_ARRAY_JOB_ID"
    )
    inference_task_id = _canonical_job_id(
        environment.get("SLURM_ARRAY_TASK_ID"), "SLURM_ARRAY_TASK_ID", allow_zero=True
    )
    training_array_job_id = _canonical_job_id(
        environment.get("VCC_TRAINING_ARRAY_JOB_ID"), "VCC_TRAINING_ARRAY_JOB_ID"
    )
    expected_dependency = f"aftercorr:{training_array_job_id}"
    require(
        environment.get("SLURM_JOB_DEPENDENCY") == expected_dependency,
        "completion environment lacks the exact aftercorr dependency",
    )
    require(runtime_slurm.get("job_id") == inference_job_id, "runtime inference job ID mismatch")
    require(
        runtime_slurm.get("array_job_id") == inference_array_job_id,
        "runtime inference array job ID mismatch",
    )
    require(
        runtime_slurm.get("array_task_id") == inference_task_id,
        "runtime inference array task ID mismatch",
    )
    require(
        runtime_slurm.get("job_dependency") == expected_dependency,
        "runtime aftercorr dependency mismatch",
    )
    require(
        runtime_slurm.get("declared_training_array_job_id") == training_array_job_id,
        "runtime training array declaration mismatch",
    )
    upstream = _mapping(runtime_slurm.get("correlated_upstream"), "correlated upstream")
    training_task_job_id = _canonical_job_id(
        upstream.get("task_job_id"), "upstream training task job ID"
    )
    require(upstream.get("array_job_id") == training_array_job_id, "upstream array job ID mismatch")
    require(upstream.get("array_task_id") == inference_task_id, "upstream array task mismatch")
    require(upstream.get("dependency_type") == "aftercorr", "upstream dependency type mismatch")

    require(selection.get("schema") == SELECTION_SCHEMA, "bad selection receipt schema")
    require(selection.get("checkpoint_layout") == "validation_step_archives", "bad selection layout")
    require(selection.get("unarchived_validation_steps") == [], "selection has unarchived steps")
    selected = validate_selected_earliest_argmin(
        selection.get("validation_results"),
        selection.get("archived_candidates"),
        selection.get("selected"),
        expected_steps=EXPECTED_VALIDATION_STEPS,
    )
    runtime_selection = _mapping(runtime.get("checkpoint_selection"), "runtime selection")
    _descriptor_matches(
        runtime_selection.get("selection_json"), selection_descriptor, "runtime selection receipt"
    )
    require(
        runtime_selection.get("selected_step") == selected["global_step"],
        "runtime selected step mismatch",
    )
    require(
        runtime_selection.get("selected_val_loss") == selected["val_loss"],
        "runtime selected loss mismatch",
    )

    require(spec.get("schema") == SPEC_SCHEMA, "bad candidate spec schema")
    require(spec.get("output_tag") == output_tag, "spec output tag mismatch")
    require(spec.get("context") == context, "spec context mismatch")
    spec_firewall = _mapping(spec.get("firewall"), "spec firewall")
    require(
        spec_firewall.get("generator_inputs_include_sealed_truth") is False,
        "spec includes sealed truth",
    )
    require(
        spec_firewall.get("treated_profiles_read_while_planning") is False,
        "spec reports treated profiles read while planning",
    )
    spec_selection = _mapping(spec.get("checkpoint_selection"), "spec checkpoint selection")
    require(spec_selection.get("global_step") == selected["global_step"], "spec selected step mismatch")
    require(spec_selection.get("val_loss") == selected["val_loss"], "spec selected loss mismatch")
    spec_inputs = _mapping(spec.get("inputs"), "spec inputs")
    _descriptor_matches(spec_inputs.get("selection_json"), selection_descriptor, "spec selection receipt")

    require(generation.get("schema") == GENERATION_SCHEMA, "bad generation report schema")
    require(
        generation.get("artifact_type") == "sealed_public_validation_prediction",
        "bad generation artifact type",
    )
    generation_configuration = _mapping(generation.get("configuration"), "generation configuration")
    require(generation_configuration.get("context") == context, "generation context mismatch")
    generation_firewall = _mapping(generation.get("data_firewall"), "generation firewall")
    require(
        generation_firewall.get("sealed_treated_profiles_read") is False,
        "generation reports sealed treated profiles read",
    )
    require(generation_firewall.get("truth_inputs_read") == [], "generation reports truth inputs")
    generation_selection = _mapping(
        generation.get("checkpoint_selection"), "generation checkpoint selection"
    )
    require(
        generation_selection.get("global_step") == selected["global_step"],
        "generation selected step mismatch",
    )
    require(
        generation_selection.get("val_loss") == selected["val_loss"],
        "generation selected loss mismatch",
    )
    generation_provenance = _mapping(generation.get("provenance"), "generation provenance")
    _descriptor_matches(
        generation_provenance.get("output_h5ad"), prediction_descriptor, "generation prediction"
    )
    _descriptor_matches(
        generation_provenance.get("checkpoint_selection"),
        selection_descriptor,
        "generation selection receipt",
    )

    require(verification.get("schema") == VERIFICATION_SCHEMA, "bad verification schema")
    require(verification.get("status") == "passed", "generation verification did not pass")
    require(verification.get("output_tag") == output_tag, "verification output tag mismatch")
    require(verification.get("context") == context, "verification context mismatch")
    require(verification.get("validation_mode") == "strict-v2", "verification is not strict-v2")
    verification_checks = _mapping(verification.get("checks"), "verification checks")
    required_checks = (
        "generator_inputs_authenticated",
        "generator_did_not_read_treated_truth",
        "generation_matches_locked_configuration",
        "prediction_hash_matches_report",
        "prediction_shape_and_groups_match_spec",
        "exact_library_and_non_state_invariants_passed",
        "strict_generator_provenance_authenticated",
    )
    require(
        all(verification_checks.get(name) is True for name in required_checks),
        "one or more strict verification checks did not pass",
    )
    verification_provenance = _mapping(
        verification.get("provenance"), "verification provenance"
    )
    _descriptor_matches(verification_provenance.get("spec"), spec_descriptor, "verified spec")
    _descriptor_matches(
        verification_provenance.get("generation_json"),
        generation_descriptor,
        "verified generation report",
    )
    _descriptor_matches(
        verification_provenance.get("prediction_h5ad"),
        prediction_descriptor,
        "verified prediction",
    )

    artifacts = {
        "inference_runtime": runtime_descriptor,
        "checkpoint_selection": selection_descriptor,
        "candidate_spec": spec_descriptor,
        "generation_report": generation_descriptor,
        "prediction_h5ad": prediction_descriptor,
        "generation_verification": verification_descriptor,
    }
    correlated_lineage = {
        "dependency": expected_dependency,
        "dependency_type": "aftercorr",
        "training_array_job_id": training_array_job_id,
        "training_array_task_id": inference_task_id,
        "training_task_job_id": training_task_job_id,
        "inference_array_job_id": inference_array_job_id,
        "inference_array_task_id": inference_task_id,
        "inference_task_job_id": inference_job_id,
        "corresponding_array_task_authenticated": True,
    }
    truth_firewall = {
        "schema": FIREWALL_SCHEMA,
        "candidate_generation_truth_blind": True,
        "sealed_treated_profiles_read": False,
        "truth_inputs_read": [],
        "scoring_invoked_by_launcher": False,
        "leaderboard_submission_authorized": False,
    }
    completion_document = {
        "schema": SCHEMA,
        "output_tag": output_tag,
        "context": context,
        "candidate_directory": str(candidate_dir),
        "correlated_slurm_lineage": correlated_lineage,
        "truth_firewall": truth_firewall,
        "artifacts": artifacts,
        "completed_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "completion_state": "complete_after_strict_generation_verification",
        "artifact_count": len(artifacts),
        "producer": producer_descriptor,
    }
    return {
        **completion_document,
        "binding": {
            "algorithm": "sha256",
            "coverage": "every_top_level_field_except_binding",
            "canonical_json_sha256": canonical_binding_sha256(completion_document),
        },
    }


def atomic_write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    require(path.parent.is_dir(), f"completion parent directory is missing: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite completion receipt: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite completion receipt: {path}") from error
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--generation", type=Path, required=True)
    parser.add_argument("--prediction", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--context", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = build_completion_payload(
        runtime_path=args.runtime,
        selection_path=args.selection,
        spec_path=args.spec,
        generation_path=args.generation,
        prediction_path=args.prediction,
        verification_path=args.verification,
        output_tag=args.output_tag,
        context=args.context,
        environment=os.environ,
        producer_path=Path(__file__).resolve(),
    )
    atomic_write_json_no_overwrite(args.output, payload)
    print(
        f"Wrote final Anvil candidate completion receipt: {args.output} "
        f"binding_sha256={payload['binding']['canonical_json_sha256']}"
    )


if __name__ == "__main__":
    main()
