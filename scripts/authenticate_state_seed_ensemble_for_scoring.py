#!/usr/bin/env python3
"""Authorize one derived STATE seed ensemble before scorer-only data is opened.

This command accepts no treated-data path.  It re-runs the ensemble verifier,
which authenticates all four transported source candidates and their files,
then records the exact plan, derived prediction, source bundle, and dedicated
scoring launcher in an immutable pre-score receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from ensemble_state_seed_candidates import (
    CONTEXT_PREFIX,
    RECEIPT_SCHEMA as ENSEMBLE_RECEIPT_SCHEMA,
    TAG_PATTERN,
    descriptor_content,
    load_json,
    regular_directory,
    regular_file,
    sha256_file,
    validate_plan,
    verify as verify_ensemble,
)


SCHEMA = "vcc-state-seed-ensemble-prescore-authentication-v1"
EXPECTED_SOURCE_COUNT = 4
REGISTERED_PLAN_IDENTITY_SHA256 = (
    "f0d4d7ce123e9fea00193c3db8fc649dc1f9ab7773a47671747693421ceb95ae"
)
REGISTERED_OUTPUT_TAG = "state_sm_anvil_reconstructed_hepg2_uniform4_mean_v1"
REGISTERED_SOURCE_TAGS = [
    "state_sm_anvil_reconstructed_hepg2_seed42_v1",
    "state_sm_anvil_reconstructed_hepg2_seed20260901_v1",
    "state_sm_anvil_reconstructed_hepg2_seed20260902_v1",
    "state_sm_anvil_reconstructed_hepg2_seed20260903_v1",
]
SOURCE_ARTIFACT_KEYS = {
    "generation",
    "prediction",
    "spec",
    "transport",
    "verification",
}
SOURCE_ARTIFACT_NAMES = {
    "generation": "generation.json",
    "prediction": "prediction.h5ad",
    "spec": "spec.json",
    "transport": "candidate_transport_verified.json",
    "verification": "generation_verified.json",
}
CONTRACT = {
    "all_four_registered_sources_required": True,
    "all_source_transport_receipts_revalidated": True,
    "all_source_files_rehashed_before_scoring": True,
    "derived_ensemble_receipt_revalidated": True,
    "derived_identity_not_single_checkpoint": True,
    "leaderboard_submission_authorized": False,
    "registered_plan_identity_locked": True,
    "sealed_treated_path_received": False,
    "treated_truth_read": False,
    "validated_before_sealed_truth_access": True,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _descriptor(path: Path) -> dict[str, Any]:
    resolved = regular_file(path, "authentication input")
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _authenticate_exact_descriptor(record: Any, label: str) -> dict[str, Any]:
    require(type(record) is dict, f"{label} descriptor is absent")
    recorded_path = record.get("path")
    require(type(recorded_path) is str and Path(recorded_path).is_absolute(),
            f"{label} descriptor path is invalid")
    resolved = regular_file(Path(recorded_path), label)
    require(str(resolved) == recorded_path, f"{label} descriptor path is not canonical")
    observed = _descriptor(resolved)
    require(
        descriptor_content(observed, f"observed {label}")
        == descriptor_content(record, f"recorded {label}"),
        f"{label} descriptor content differs",
    )
    return observed


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(type(left) is dict and type(right) is dict, f"{label} descriptor is absent")
    require(left.get("path") == right.get("path"), f"{label} path differs")
    require(
        descriptor_content(left, f"{label} left")
        == descriptor_content(right, f"{label} right"),
        f"{label} content differs",
    )


def _bundle_digest(sources: list[dict[str, Any]]) -> str:
    identity = [
        {"tag": source["tag"], "artifacts": source["artifacts"]}
        for source in sources
    ]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _registered_plan(plan: dict[str, Any]) -> dict[str, Any]:
    validated = validate_plan(plan)
    identity = hashlib.sha256(
        json.dumps(validated, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    require(identity == REGISTERED_PLAN_IDENTITY_SHA256,
            "Ensemble plan differs from the independently registered identity")
    require(validated["context"] == "HepG2", "Registered context differs")
    require(validated["output_tag"] == REGISTERED_OUTPUT_TAG,
            "Registered ensemble output tag differs")
    require(validated["source_tags"] == REGISTERED_SOURCE_TAGS,
            "Registered source tags or order differ")
    require(validated["axes"] == {
        "cells_per_target": 400,
        "contexts": 1,
        "genes": 18533,
        "targets": 300,
    }, "Registered ensemble axes differ")
    require(validated["method"]["tie_break_seed"] == 20260901,
            "Registered ensemble tie-break seed differs")
    return validated


def _atomic_new_json(path: Path, payload: dict[str, Any]) -> None:
    require(path.suffix == ".json", "Pre-score receipt must end in .json")
    require(not path.exists() and not path.is_symlink(), f"Refusing to overwrite: {path}")
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
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _expected_receipt_path(public_root: Path, context: str, output_tag: str) -> Path:
    return (
        public_root
        / "ensemble_scoring"
        / CONTEXT_PREFIX[context]
        / output_tag
        / "ensemble_prescore_authenticated.json"
    )


def build_payload(
    *,
    project_dir: Path,
    public_root: Path,
    plan_path: Path,
    ensemble_run_dir: Path,
    launcher_path: Path,
) -> dict[str, Any]:
    project_dir = regular_directory(project_dir, "project directory")
    public_root = regular_directory(public_root, "public root")
    plan_path = regular_file(plan_path, "ensemble plan")
    launcher_path = regular_file(launcher_path, "ensemble scoring launcher")
    require(plan_path == (project_dir / "configs/state/anvil_seed_ensemble_v1.json").resolve(),
            "Ensemble plan path is not the registered project path")
    require(launcher_path
            == (project_dir / "slurm/cpu_score_public_state_seed_ensemble_v1.sbatch").resolve(),
            "Ensemble scoring launcher is not the registered project path")
    expected_public_root = (project_dir / "artifacts" / "anvil_state_reconstruction").resolve(
        strict=True
    )
    require(public_root == expected_public_root,
            "Public root differs from the registered local artifact root")
    plan = _registered_plan(load_json(plan_path, "ensemble plan"))
    require(len(plan["source_tags"]) == EXPECTED_SOURCE_COUNT,
            "The scoring lane requires exactly four registered sources")
    ensemble_run_dir = regular_directory(ensemble_run_dir, "ensemble run directory")
    verified = verify_ensemble(
        project_dir=project_dir,
        public_root=public_root,
        plan_path=plan_path,
        run_dir=ensemble_run_dir,
    )
    require(
        verified.get("schema") == ENSEMBLE_RECEIPT_SCHEMA
        and verified.get("status") == "passed",
        "Derived ensemble verification did not pass",
    )
    require(
        verified.get("lineage")
        == "truth-blind-derived-seed-ensemble-not-historical-p4",
        "Ensemble lineage is not the registered derived lineage",
    )
    require(
        verified.get("evaluation_policy") == plan["evaluation_policy"],
        "Ensemble evaluation policy differs from the registered plan",
    )
    expected_run = (
        public_root
        / "ensembles"
        / CONTEXT_PREFIX[plan["context"]]
        / plan["output_tag"]
    ).resolve(strict=True)
    require(ensemble_run_dir == expected_run, "Ensemble run directory differs from plan")
    require(ensemble_run_dir.is_relative_to(public_root),
            "Ensemble run directory escapes the public root")
    ensemble_receipt_path = regular_file(
        ensemble_run_dir / "ensemble_verified.json", "ensemble verification receipt"
    )
    prediction_path = regular_file(
        ensemble_run_dir / "prediction.h5ad", "ensemble prediction"
    )
    ensemble_receipt_descriptor = _descriptor(ensemble_receipt_path)
    prediction_descriptor = _descriptor(prediction_path)
    _same_descriptor(prediction_descriptor, verified.get("output"), "ensemble prediction")

    recorded_sources = verified.get("sources")
    require(
        type(recorded_sources) is list
        and len(recorded_sources) == EXPECTED_SOURCE_COUNT,
        "Ensemble receipt does not bind four source candidates",
    )
    sources: list[dict[str, Any]] = []
    prefix = CONTEXT_PREFIX[plan["context"]]
    for expected_tag, recorded in zip(plan["source_tags"], recorded_sources, strict=True):
        require(type(recorded) is dict and recorded.get("tag") == expected_tag,
                "Ensemble source order or tag differs from the plan")
        expected_run_dir = (
            public_root / "candidates" / prefix / expected_tag
        ).resolve(strict=True)
        require(
            Path(str(recorded.get("run_directory", ""))).resolve(strict=True)
            == expected_run_dir,
            f"Source run directory differs: {expected_tag}",
        )
        require(expected_run_dir.is_relative_to(public_root),
                f"Source run directory escapes the public root: {expected_tag}")
        artifacts = recorded.get("artifacts")
        require(type(artifacts) is dict and set(artifacts) == SOURCE_ARTIFACT_KEYS,
                f"Source artifact fields differ: {expected_tag}")
        authenticated = {}
        for key in sorted(SOURCE_ARTIFACT_KEYS):
            expected_artifact = (expected_run_dir / SOURCE_ARTIFACT_NAMES[key]).resolve(
                strict=True
            )
            require(Path(str(artifacts[key].get("path", ""))) == expected_artifact,
                    f"Source artifact path differs: {expected_tag}/{key}")
            authenticated[key] = _authenticate_exact_descriptor(
                artifacts[key], f"{expected_tag}/{key}"
            )
        sources.append(
            {
                "tag": expected_tag,
                "run_directory": str(expected_run_dir),
                "artifacts": authenticated,
            }
        )

    first_spec_path = Path(sources[0]["artifacts"]["spec"]["path"])
    first_spec = load_json(first_spec_path, "first ensemble source spec")
    panel = (first_spec.get("inputs") or {}).get("panel_csv")
    authenticated_panel = _authenticate_exact_descriptor(panel, "shared panel CSV")
    canonical_manifest = (project_dir / "dataset" / "public_v51" / "split_manifest.csv").resolve()
    require(Path(authenticated_panel["path"]) == canonical_manifest,
            "Ensemble sources do not bind the canonical local panel manifest")
    return {
        "schema": SCHEMA,
        "status": "passed",
        "identity_kind": "derived-uniform-state-seed-ensemble",
        "lineage": "truth-blind-derived-seed-ensemble-not-historical-p4",
        "context": plan["context"],
        "output_tag": plan["output_tag"],
        "evaluation_policy": plan["evaluation_policy"],
        "plan": _descriptor(plan_path),
        "registered_plan_identity_sha256": REGISTERED_PLAN_IDENTITY_SHA256,
        "ensemble": {
            "run_directory": str(ensemble_run_dir),
            "verification": ensemble_receipt_descriptor,
            "prediction": prediction_descriptor,
        },
        "source_candidates": sources,
        "source_bundle_identity_sha256": _bundle_digest(sources),
        "shared_inputs": {"panel_csv": authenticated_panel},
        "implementation": {
            "authenticator": _descriptor(Path(__file__).resolve()),
            "ensemble_builder": _descriptor(
                project_dir / "scripts" / "ensemble_state_seed_candidates.py"
            ),
            "scoring_launcher": _descriptor(launcher_path),
        },
        "contract": dict(CONTRACT),
    }


def validate_receipt(
    path: Path,
    *,
    expected_context: str,
    expected_output_tag: str,
    plan_path: Path,
    ensemble_run_dir: Path,
    rehash_sources: bool,
) -> dict[str, Any]:
    require(expected_context in CONTEXT_PREFIX, "Unsupported ensemble context")
    require(TAG_PATTERN.fullmatch(expected_output_tag) is not None, "Unsafe output tag")
    payload = load_json(path, "ensemble pre-score receipt")
    expected_fields = {
        "context",
        "contract",
        "ensemble",
        "evaluation_policy",
        "identity_kind",
        "implementation",
        "lineage",
        "output_tag",
        "plan",
        "registered_plan_identity_sha256",
        "schema",
        "shared_inputs",
        "source_bundle_identity_sha256",
        "source_candidates",
        "status",
    }
    require(set(payload) == expected_fields, "Pre-score receipt fields are not exact")
    require(payload.get("schema") == SCHEMA and payload.get("status") == "passed",
            "Pre-score receipt is not a passed v1 receipt")
    require(payload.get("identity_kind") == "derived-uniform-state-seed-ensemble",
            "Pre-score receipt mislabels the candidate identity")
    require(
        payload.get("lineage") == "truth-blind-derived-seed-ensemble-not-historical-p4",
        "Pre-score receipt lineage differs",
    )
    require(payload.get("context") == expected_context
            and payload.get("output_tag") == expected_output_tag,
            "Pre-score receipt identity differs from the score request")
    plan_path = regular_file(plan_path, "ensemble plan")
    plan = _registered_plan(load_json(plan_path, "ensemble plan"))
    require(
        payload.get("registered_plan_identity_sha256")
        == REGISTERED_PLAN_IDENTITY_SHA256,
        "Pre-score registered-plan identity differs",
    )
    require(plan["context"] == expected_context and plan["output_tag"] == expected_output_tag,
            "Ensemble plan identity differs from the score request")
    require(payload.get("evaluation_policy") == plan["evaluation_policy"],
            "Pre-score evaluation policy differs from plan")
    _same_descriptor(_descriptor(plan_path), payload.get("plan"), "ensemble plan")
    ensemble_run_dir = regular_directory(ensemble_run_dir, "ensemble run directory")
    public_root = ensemble_run_dir.parents[2]
    project_dir = public_root.parents[1]
    require(
        public_root == (project_dir / "artifacts/anvil_state_reconstruction").resolve(),
        "Pre-score ensemble public root is not canonical",
    )
    require(
        plan_path == (project_dir / "configs/state/anvil_seed_ensemble_v1.json").resolve(),
        "Pre-score plan path is not canonical",
    )
    require(
        regular_file(path, "ensemble pre-score receipt")
        == _expected_receipt_path(
            public_root, expected_context, expected_output_tag
        ).resolve(),
        "Pre-score receipt path is not canonical",
    )
    ensemble = payload.get("ensemble")
    require(type(ensemble) is dict and set(ensemble) == {
        "prediction", "run_directory", "verification"
    }, "Pre-score ensemble fields are not exact")
    require(Path(str(ensemble["run_directory"])).resolve(strict=True) == ensemble_run_dir,
            "Pre-score ensemble directory differs")
    _same_descriptor(
        _descriptor(ensemble_run_dir / "ensemble_verified.json"),
        ensemble["verification"],
        "ensemble verification",
    )
    _same_descriptor(
        _descriptor(ensemble_run_dir / "prediction.h5ad"),
        ensemble["prediction"],
        "ensemble prediction",
    )
    sources = payload.get("source_candidates")
    require(type(sources) is list and len(sources) == EXPECTED_SOURCE_COUNT,
            "Pre-score receipt does not contain four sources")
    require([source.get("tag") for source in sources] == plan["source_tags"],
            "Pre-score source tags differ from plan")
    prefix = CONTEXT_PREFIX[expected_context]
    for source in sources:
        require(type(source) is dict and set(source) == {
            "artifacts", "run_directory", "tag"
        }, "Pre-score source fields are not exact")
        artifacts = source.get("artifacts")
        require(type(artifacts) is dict and set(artifacts) == SOURCE_ARTIFACT_KEYS,
                "Pre-score source artifact fields differ")
        expected_source_run = (
            ensemble_run_dir.parents[2]
            / "candidates"
            / prefix
            / source["tag"]
        ).resolve(strict=True)
        require(
            Path(str(source.get("run_directory", ""))).resolve(strict=True)
            == expected_source_run,
            f"Pre-score source directory differs: {source['tag']}",
        )
        require(expected_source_run.is_relative_to(ensemble_run_dir.parents[2]),
                f"Pre-score source directory escapes public root: {source['tag']}")
        for key in SOURCE_ARTIFACT_KEYS:
            expected_artifact = (expected_source_run / SOURCE_ARTIFACT_NAMES[key]).resolve(
                strict=True
            )
            require(Path(str(artifacts[key].get("path", ""))) == expected_artifact,
                    f"Pre-score source artifact path differs: {source['tag']}/{key}")
    require(payload.get("source_bundle_identity_sha256") == _bundle_digest(sources),
            "Pre-score source-bundle digest differs")
    if rehash_sources:
        for source in sources:
            artifacts = source.get("artifacts")
            for key in SOURCE_ARTIFACT_KEYS:
                _authenticate_exact_descriptor(artifacts[key], f"{source['tag']}/{key}")
    shared_inputs = payload.get("shared_inputs")
    require(type(shared_inputs) is dict and set(shared_inputs) == {"panel_csv"},
            "Pre-score shared-input fields are not exact")
    _authenticate_exact_descriptor(shared_inputs["panel_csv"], "shared panel CSV")
    first_spec = load_json(
        Path(sources[0]["artifacts"]["spec"]["path"]), "first ensemble source spec"
    )
    _same_descriptor(
        shared_inputs["panel_csv"],
        (first_spec.get("inputs") or {}).get("panel_csv"),
        "source/shared panel CSV",
    )
    implementation = payload.get("implementation")
    require(type(implementation) is dict and set(implementation) == {
        "authenticator", "ensemble_builder", "scoring_launcher"
    }, "Pre-score implementation fields are not exact")
    for key, expected_name in (
        ("authenticator", "authenticate_state_seed_ensemble_for_scoring.py"),
        ("ensemble_builder", "ensemble_state_seed_candidates.py"),
        ("scoring_launcher", "cpu_score_public_state_seed_ensemble_v1.sbatch"),
    ):
        descriptor = implementation[key]
        require(Path(str(descriptor.get("path", ""))).name == expected_name,
                f"Pre-score implementation name differs: {key}")
        _authenticate_exact_descriptor(descriptor, f"implementation/{key}")
    require(payload.get("contract") == CONTRACT, "Pre-score contract is not exact")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--project-dir", type=Path, required=True)
    build_parser.add_argument("--public-root", type=Path, required=True)
    build_parser.add_argument("--plan", type=Path, required=True)
    build_parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    build_parser.add_argument("--scoring-launcher", type=Path, required=True)
    build_parser.add_argument("--output-json", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--receipt", type=Path, required=True)
    verify_parser.add_argument("--context", choices=tuple(CONTEXT_PREFIX), required=True)
    verify_parser.add_argument("--output-tag", required=True)
    verify_parser.add_argument("--plan", type=Path, required=True)
    verify_parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    verify_parser.add_argument("--rehash-sources", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.command == "build":
        payload = build_payload(
            project_dir=args.project_dir,
            public_root=args.public_root,
            plan_path=args.plan,
            ensemble_run_dir=args.ensemble_run_dir,
            launcher_path=args.scoring_launcher,
        )
        public_root = regular_directory(args.public_root, "public root")
        expected_output = _expected_receipt_path(
            public_root, payload["context"], payload["output_tag"]
        )
        require(
            args.output_json.absolute() == expected_output,
            "Pre-score receipt output path is not canonical",
        )
        require(
            regular_directory(args.output_json.parent, "pre-score receipt directory")
            == expected_output.parent.resolve(),
            "Pre-score receipt directory is not canonical",
        )
        _atomic_new_json(args.output_json, payload)
    else:
        payload = validate_receipt(
            args.receipt,
            expected_context=args.context,
            expected_output_tag=args.output_tag,
            plan_path=args.plan,
            ensemble_run_dir=args.ensemble_run_dir,
            rehash_sources=args.rehash_sources,
        )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
