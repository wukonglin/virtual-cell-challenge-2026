#!/usr/bin/env python3
"""Authenticate one cell-eval2 public-validation scoring run.

The validator understands the VCC 2026 aggregation contract: normalized MSE
is a derived ratio of sums with no per-target result rows, direction metrics
may contain legitimate NaNs, and NMAE may omit a real-side-only target subset.
It binds those semantics to the scorer sidecars and pinned software identity.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import tomllib
from pathlib import Path
from typing import Any

import cell_eval2
import numpy as np
import pandas as pd

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, require
from public_candidate_v6_contract import (
    SPEC_SCHEMA,
    VERIFICATION_SCHEMA,
    _validate_strict_spec_provenance,
    validate_output_tag,
)
from summarize_public_validation import (
    FIDELITY_METRIC,
    MSE_METRIC,
    NMAE_METRIC,
    REACH_METRIC,
    SCORED_METRICS,
    _mean_summary,
    _read_results,
    _validate_cell_eval_sidecars,
)


SCHEMA = "vcc-public-cell-eval-run-verification-v1"
EXPECTED_CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"
EXPECTED_CELL_EVAL_VERSION = "0.16.0"
EXPECTED_PDEX_VERSION = "0.3.0"
VIEWS_SCHEMA = "vcc-public-scoring-views-v1"
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


def _same_descriptor(left: Any, right: Any, label: str) -> None:
    require(isinstance(left, dict) and isinstance(right, dict), f"{label} descriptor is absent")
    for field in ("path", "size_bytes", "sha256"):
        require(left.get(field) == right.get(field), f"{label} descriptor differs: {field}")
    if "mtime_ns" in left:
        require(left.get("mtime_ns") == right.get("mtime_ns"), f"{label} descriptor differs: mtime_ns")


def _authenticate_view_output(record: Any, path: Path, label: str) -> dict[str, Any]:
    require(isinstance(record, dict), f"{label} descriptor is absent")
    observed = describe_file(path)
    _same_descriptor(record, observed, label)
    return observed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", choices=("HepG2", "Jurkat"), required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-spec", type=Path, required=True)
    parser.add_argument("--generation-verification", type=Path, required=True)
    parser.add_argument("--views-json", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--cell-eval-checkout", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_CELL_EVAL_COMMIT)
    parser.add_argument("--expected-version", default=EXPECTED_CELL_EVAL_VERSION)
    parser.add_argument("--expected-pdex", default=EXPECTED_PDEX_VERSION)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _git(checkout: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_software(
    checkout: Path, expected_commit: str, expected_version: str, expected_pdex: str
) -> dict[str, Any]:
    resolved = checkout.resolve()
    require((resolved / ".git").is_dir(), "cell-eval2 checkout lacks Git metadata")
    commit = _git(resolved, "rev-parse", "HEAD")
    require(commit == expected_commit, "cell-eval2 commit differs from the scoring pin")
    require(
        _git(resolved, "status", "--porcelain", "--untracked-files=normal") == "",
        "cell-eval2 checkout is dirty",
    )
    module_path = Path(cell_eval2.__file__).resolve()
    require(
        module_path.is_relative_to(resolved / "src"),
        "cell_eval2 imported outside the pinned checkout",
    )
    with (resolved / "pyproject.toml").open("rb") as handle:
        declared_version = tomllib.load(handle)["project"]["version"]
    pdex_version = importlib.metadata.version("pdex")
    require(declared_version == expected_version, "Unexpected cell-eval2 declared version")
    require(pdex_version == expected_pdex, "Unexpected pdex version")
    return {
        "cell_eval2_git_commit": commit,
        "cell_eval2_checkout_clean": True,
        "cell_eval2_declared_version": declared_version,
        "cell_eval2_runtime_version": cell_eval2.__version__,
        "cell_eval2_module": str(module_path),
        "pdex_version": pdex_version,
    }


def validate_run_meta(path: Path, expected_pdex: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(payload.get("resolved_device") == "cpu", "cell-eval2 did not resolve to CPU")
    require(payload.get("resolved_de_backend") == "pdex", "cell-eval2 did not resolve to pdex")
    require(payload.get("source_fingerprint_strict") is True, "Strict scorer fingerprinting is disabled")
    require(payload.get("input_type_real_effective") == "counts", "Real input is not counts")
    require(payload.get("input_type_pred_effective") == "counts", "Prediction input is not counts")
    require(payload.get("comparator") == "bulk_lognorm", "Unexpected VCC comparator")
    pdex = (payload.get("environment") or {}).get("pdex") or {}
    require(pdex.get("version") == expected_pdex, "run_meta has the wrong pdex version")
    digest = payload.get("config_digest")
    require(isinstance(digest, str) and bool(digest), "run_meta lacks config_digest")
    return {
        "config_digest": digest,
        "resolved_device": payload["resolved_device"],
        "resolved_de_backend": payload["resolved_de_backend"],
        "comparator": payload["comparator"],
        "source_fingerprint_strict": True,
        "pdex_version": pdex["version"],
    }


def validate_candidate_receipts(
    spec_path: Path,
    verification_path: Path,
    views_path: Path,
    run_meta_path: Path,
    manifest_path: Path,
    *,
    context: str,
    output_tag: str,
) -> dict[str, Any]:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    views = json.loads(views_path.read_text(encoding="utf-8"))
    require(spec.get("schema") == SPEC_SCHEMA, "Bad candidate spec schema")
    require(verification.get("schema") == VERIFICATION_SCHEMA, "Bad generation verification schema")
    require(verification.get("status") == "passed", "Generation verification did not pass")
    require(
        spec.get("context") == verification.get("context") == views.get("context") == context,
        "Candidate receipt contexts differ",
    )
    require(
        spec.get("output_tag") == verification.get("output_tag") == output_tag,
        "Candidate receipt tags differ",
    )
    checks = verification.get("checks")
    require(
        isinstance(checks, dict) and bool(checks),
        "Generation verification checks are missing or empty",
    )
    require(
        all(value is True for value in checks.values()),
        "Generation verification has a failed check",
    )
    _validate_strict_spec_provenance(spec)
    _same_descriptor(
        (spec.get("inputs") or {}).get("panel_csv"),
        describe_file(manifest_path),
        "Candidate panel manifest",
    )
    require(
        checks.get("strict_generator_provenance_authenticated") is True,
        "Strict generator provenance was not authenticated",
    )
    require(
        verification.get("validation_mode") == "strict-v2",
        "Generation verification is not in strict-v2 mode",
    )
    require(
        verification.get("configuration") == spec.get("configuration"),
        "Generation verification configuration differs from the spec",
    )
    generation_provenance = verification.get("provenance")
    require(isinstance(generation_provenance, dict), "Generation provenance is absent")
    _same_descriptor(
        generation_provenance.get("spec"),
        describe_file(spec_path),
        "Generation spec",
    )
    require(set(views) == VIEWS_FIELDS, "Scoring views fields are not exact")
    require(views.get("schema") == VIEWS_SCHEMA, "Bad scoring views schema")
    contract = views.get("contract", {})
    require(contract.get("model_read_treated_truth") is False, "Scoring views violate the model firewall")
    require(contract.get("input_type") == "raw-nonnegative-integer-counts", "Views are not raw counts")
    provenance = views.get("provenance")
    require(
        isinstance(provenance, dict) and set(provenance) == VIEWS_PROVENANCE_FIELDS,
        "Scoring views provenance fields are not exact",
    )
    _same_descriptor(
        provenance.get("prediction"),
        generation_provenance.get("prediction_h5ad"),
        "Scoring-view prediction",
    )
    expected_prediction_view = (views_path.parent / "prediction_view.h5ad").resolve()
    expected_truth_view = (views_path.parent / "truth_view.h5ad").resolve()
    require(
        Path(str((provenance.get("output_prediction") or {}).get("path", ""))).resolve()
        == expected_prediction_view,
        "Prediction-view output path differs from the score directory",
    )
    require(
        Path(str((provenance.get("output_truth") or {}).get("path", ""))).resolve()
        == expected_truth_view,
        "Truth-view output path differs from the score directory",
    )
    authenticated_prediction_view = _authenticate_view_output(
        provenance.get("output_prediction"),
        expected_prediction_view,
        "Prediction-view output",
    )
    authenticated_truth_view = _authenticate_view_output(
        provenance.get("output_truth"),
        expected_truth_view,
        "Truth-view output",
    )
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    require(isinstance(run_meta, dict), "run_meta must be a JSON object")
    require(
        Path(str(run_meta.get("source", ""))).resolve() == expected_truth_view,
        "run_meta truth source differs from the authenticated truth view",
    )
    return {
        "spec": describe_file(spec_path),
        "generation_verification": describe_file(verification_path),
        "views": describe_file(views_path),
        "prediction_view": authenticated_prediction_view,
        "truth_view": authenticated_truth_view,
    }


def validate_metric_outputs(
    results_path: Path, manifest_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = pd.read_csv(manifest_path, dtype=str, keep_default_na=False)
    require("target_gene" in manifest.columns, "Manifest lacks target_gene")
    targets = manifest["target_gene"].astype(str)
    require(len(targets) == 300 and targets.is_unique, "Manifest target axis is invalid")
    target_set = set(targets)
    raw_results = pd.read_csv(results_path)
    require("metric" in raw_results.columns, "results.csv lacks metric")
    require(
        not (raw_results["metric"].astype(str) == MSE_METRIC).any(),
        "Derived MSE must not appear as per-target rows",
    )
    frame = _read_results(results_path, target_set)
    summary = _mean_summary(frame, len(target_set))
    sidecars = _validate_cell_eval_sidecars(
        results_path, frame, summary, len(target_set)
    )
    metric_qc: dict[str, Any] = {}
    for metric in SCORED_METRICS:
        record = summary[metric]
        metric_qc[metric] = {
            "aggregate": record["mean"],
            "finite_targets": int(record["finite_targets"]),
            "missing_targets": int(record["missing_targets"]),
            "aggregation": record["aggregation"],
        }
    require(
        metric_qc[MSE_METRIC]["aggregation"] == "ratio_of_sums"
        and metric_qc[MSE_METRIC]["finite_targets"] == 300,
        "Derived MSE was not authenticated from complete component axes",
    )
    require(metric_qc[NMAE_METRIC]["finite_targets"] > 0, "NMAE has no usable targets")
    for metric in (FIDELITY_METRIC, REACH_METRIC):
        require(metric_qc[metric]["finite_targets"] > 0, f"{metric} has no finite targets")
    return metric_qc, sidecars


def main() -> None:
    args = parse_args()
    validate_output_tag(args.output_tag)
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    for path in (
        args.manifest,
        args.candidate_spec,
        args.generation_verification,
        args.views_json,
        args.results_csv,
        args.run_meta,
    ):
        require(path.is_file(), f"Missing scorer verification input: {path}")
    software = validate_software(
        args.cell_eval_checkout,
        args.expected_commit,
        args.expected_version,
        args.expected_pdex,
    )
    candidate = validate_candidate_receipts(
        args.candidate_spec,
        args.generation_verification,
        args.views_json,
        args.run_meta,
        args.manifest,
        context=args.context,
        output_tag=args.output_tag,
    )
    run_meta = validate_run_meta(args.run_meta, args.expected_pdex)
    metrics, sidecars = validate_metric_outputs(args.results_csv, args.manifest)
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "context": args.context,
        "output_tag": args.output_tag,
        "software": software,
        "resolved_run": run_meta,
        "metrics": metrics,
        "candidate_receipts": candidate,
        "outputs": {
            "results": describe_file(args.results_csv),
            "run_meta": describe_file(args.run_meta),
            **sidecars,
        },
        "contract": {
            "derived_mse_reconstructed_from_complete_components": True,
            "aggregate_mse_matches_cell_eval2": True,
            "metric_aggregation_n_used_authenticated": True,
            "direction_nan_targets_are_retained_and_counted": True,
            "nmae_real_side_omission_is_explicit": True,
        },
    }
    atomic_write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
