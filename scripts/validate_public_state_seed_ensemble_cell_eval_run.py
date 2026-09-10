#!/usr/bin/env python3
"""Authenticate cell-eval2 output for a derived STATE seed ensemble.

Unlike the single-checkpoint validator, this receipt binds a pre-score
ensemble authorization and never accepts a candidate spec or generation
verification.  Metric and scorer checks remain identical to the pinned public
validation lane.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from authenticate_state_seed_ensemble_for_scoring import (
    SCHEMA as PRESCORE_SCHEMA,
    validate_receipt as validate_prescore_receipt,
)
from ensemble_state_seed_candidates import (
    RECEIPT_SCHEMA as ENSEMBLE_RECEIPT_SCHEMA,
    load_json,
    validate_plan,
)
from validate_public_cell_eval_run import (
    EXPECTED_CELL_EVAL_COMMIT,
    EXPECTED_CELL_EVAL_VERSION,
    EXPECTED_PDEX_VERSION,
    VIEWS_FIELDS,
    VIEWS_PROVENANCE_FIELDS,
    VIEWS_SCHEMA,
    _authenticate_view_output,
    _same_descriptor,
    atomic_write_json,
    describe_file,
    require,
    validate_metric_outputs,
    validate_run_meta,
    validate_software,
)


SCHEMA = "vcc-public-state-seed-ensemble-cell-eval-run-verification-v1"
CONTRACT = {
    "aggregate_mse_matches_cell_eval2": True,
    "derived_identity_not_single_checkpoint": True,
    "derived_mse_reconstructed_from_complete_components": True,
    "direction_nan_targets_are_retained_and_counted": True,
    "ensemble_authenticated_before_treated_truth_access": True,
    "metric_aggregation_n_used_authenticated": True,
    "nmae_real_side_omission_is_explicit": True,
    "pre_registered_comparison_gate_preserved": True,
}


def validate_ensemble_receipts(
    *,
    context: str,
    output_tag: str,
    manifest_path: Path,
    plan_path: Path,
    ensemble_verification_path: Path,
    prescore_path: Path,
    ensemble_run_dir: Path,
    views_path: Path,
    run_meta_path: Path,
) -> dict[str, Any]:
    plan = validate_plan(load_json(plan_path, "ensemble plan"))
    require(plan["context"] == context and plan["output_tag"] == output_tag,
            "Ensemble plan identity differs from the score request")
    ensemble = load_json(ensemble_verification_path, "ensemble verification")
    require(
        ensemble.get("schema") == ENSEMBLE_RECEIPT_SCHEMA
        and ensemble.get("status") == "passed",
        "Ensemble verification is not a passed derived receipt",
    )
    require(
        ensemble.get("lineage")
        == "truth-blind-derived-seed-ensemble-not-historical-p4",
        "Ensemble verification lineage differs",
    )
    require(ensemble.get("context") == context and ensemble.get("output_tag") == output_tag,
            "Ensemble verification identity differs")
    require(ensemble.get("evaluation_policy") == plan["evaluation_policy"],
            "Ensemble verification changed the registered evaluation policy")

    prescore = validate_prescore_receipt(
        prescore_path,
        expected_context=context,
        expected_output_tag=output_tag,
        plan_path=plan_path,
        ensemble_run_dir=ensemble_run_dir,
        rehash_sources=False,
    )
    require(prescore.get("schema") == PRESCORE_SCHEMA,
            "Bad ensemble pre-score receipt schema")
    _same_descriptor(
        prescore["ensemble"]["verification"],
        describe_file(ensemble_verification_path),
        "Pre-score ensemble verification",
    )

    views = load_json(views_path, "ensemble scoring views")
    require(set(views) == VIEWS_FIELDS, "Scoring views fields are not exact")
    require(views.get("schema") == VIEWS_SCHEMA, "Bad scoring views schema")
    require(views.get("context") == context, "Scoring views context differs")
    axes = views.get("axes") or {}
    require(
        axes.get("genes") == 7107
        and axes.get("targets") == plan["axes"]["targets"]
        and axes.get("prediction_treated_cells")
        == plan["axes"]["targets"] * plan["axes"]["cells_per_target"],
        "Scoring-view axes differ from the registered ensemble plan",
    )
    view_contract = views.get("contract") or {}
    require(view_contract.get("model_read_treated_truth") is False,
            "Scoring views violate the model firewall")
    require(view_contract.get("input_type") == "raw-nonnegative-integer-counts",
            "Scoring views are not raw counts")
    provenance = views.get("provenance")
    require(type(provenance) is dict and set(provenance) == VIEWS_PROVENANCE_FIELDS,
            "Scoring views provenance fields are not exact")
    _same_descriptor(
        prescore["ensemble"]["prediction"],
        provenance.get("prediction"),
        "Scoring-view ensemble prediction",
    )
    _same_descriptor(
        prescore["shared_inputs"]["panel_csv"],
        describe_file(manifest_path),
        "Scoring manifest",
    )
    expected_prediction_view = (views_path.parent / "prediction_view.h5ad").resolve()
    expected_truth_view = (views_path.parent / "truth_view.h5ad").resolve()
    require(
        Path(str((provenance.get("output_prediction") or {}).get("path", ""))).resolve()
        == expected_prediction_view,
        "Prediction-view path differs from the ensemble score directory",
    )
    require(
        Path(str((provenance.get("output_truth") or {}).get("path", ""))).resolve()
        == expected_truth_view,
        "Truth-view path differs from the ensemble score directory",
    )
    prediction_view = _authenticate_view_output(
        provenance.get("output_prediction"), expected_prediction_view,
        "Ensemble prediction-view output"
    )
    truth_view = _authenticate_view_output(
        provenance.get("output_truth"), expected_truth_view,
        "Ensemble truth-view output"
    )
    run_meta = load_json(run_meta_path, "ensemble cell-eval run metadata")
    require(Path(str(run_meta.get("source", ""))).resolve() == expected_truth_view,
            "run_meta truth source differs from the authenticated truth view")

    # This timestamp check complements the launcher's strict command ordering:
    # the immutable pre-score receipt must exist before any truth-bearing view.
    prescore_mtime = prescore_path.stat().st_mtime_ns
    require(
        views_path.stat().st_mtime_ns >= prescore_mtime
        and expected_prediction_view.stat().st_mtime_ns >= prescore_mtime
        and expected_truth_view.stat().st_mtime_ns >= prescore_mtime,
        "A scoring view predates ensemble pre-score authentication",
    )
    return {
        "plan": describe_file(plan_path),
        "ensemble_verification": describe_file(ensemble_verification_path),
        "prescore_authentication": describe_file(prescore_path),
        "prediction": prescore["ensemble"]["prediction"],
        "views": describe_file(views_path),
        "prediction_view": prediction_view,
        "truth_view": truth_view,
        "source_bundle_identity_sha256": prescore["source_bundle_identity_sha256"],
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", choices=("HepG2",), required=True)
    parser.add_argument("--output-tag", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--ensemble-plan", type=Path, required=True)
    parser.add_argument("--ensemble-verification", type=Path, required=True)
    parser.add_argument("--prescore-authentication", type=Path, required=True)
    parser.add_argument("--ensemble-run-dir", type=Path, required=True)
    parser.add_argument("--views-json", type=Path, required=True)
    parser.add_argument("--results-csv", type=Path, required=True)
    parser.add_argument("--run-meta", type=Path, required=True)
    parser.add_argument("--cell-eval-checkout", type=Path, required=True)
    parser.add_argument("--expected-commit", default=EXPECTED_CELL_EVAL_COMMIT)
    parser.add_argument("--expected-version", default=EXPECTED_CELL_EVAL_VERSION)
    parser.add_argument("--expected-pdex", default=EXPECTED_PDEX_VERSION)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    require(not args.output_json.exists(), f"Refusing to overwrite: {args.output_json}")
    for path in (
        args.manifest,
        args.ensemble_plan,
        args.ensemble_verification,
        args.prescore_authentication,
        args.views_json,
        args.results_csv,
        args.run_meta,
    ):
        require(path.is_file(), f"Missing ensemble scorer verification input: {path}")
    software = validate_software(
        args.cell_eval_checkout,
        args.expected_commit,
        args.expected_version,
        args.expected_pdex,
    )
    receipts = validate_ensemble_receipts(
        context=args.context,
        output_tag=args.output_tag,
        manifest_path=args.manifest,
        plan_path=args.ensemble_plan,
        ensemble_verification_path=args.ensemble_verification,
        prescore_path=args.prescore_authentication,
        ensemble_run_dir=args.ensemble_run_dir,
        views_path=args.views_json,
        run_meta_path=args.run_meta,
    )
    resolved_run = validate_run_meta(args.run_meta, args.expected_pdex)
    metrics, sidecars = validate_metric_outputs(args.results_csv, args.manifest)
    payload = {
        "schema": SCHEMA,
        "status": "passed",
        "identity_kind": "derived-uniform-state-seed-ensemble",
        "lineage": "truth-blind-derived-seed-ensemble-not-historical-p4",
        "context": args.context,
        "output_tag": args.output_tag,
        "evaluation_policy": validate_plan(
            load_json(args.ensemble_plan, "ensemble plan")
        )["evaluation_policy"],
        "software": software,
        "resolved_run": resolved_run,
        "metrics": metrics,
        "ensemble_receipts": receipts,
        "outputs": {
            "results": describe_file(args.results_csv),
            "run_meta": describe_file(args.run_meta),
            **sidecars,
        },
        "contract": dict(CONTRACT),
    }
    atomic_write_json(args.output_json, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
