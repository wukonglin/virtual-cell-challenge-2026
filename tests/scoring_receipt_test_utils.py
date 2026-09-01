"""Small authenticated scoring receipts for comparator unit tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


CONFIG_DIGEST = "1" * 64
SOURCE_FINGERPRINT = "2" * 64
ANCHOR_IDENTITY = "3" * 64
CELL_EVAL_COMMIT = "5e64833518a6603a0301cbe28185d49c30f4a986"


def descriptor(path: Path, *, mtime: bool = True) -> dict[str, Any]:
    stat = path.stat()
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "size_bytes": stat.st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    if mtime:
        record["mtime_ns"] = stat.st_mtime_ns
    return record


def software() -> dict[str, Any]:
    return {
        "cell_eval2_git_commit": CELL_EVAL_COMMIT,
        "cell_eval2_checkout_clean": True,
        "cell_eval2_declared_version": "0.16.0",
        "cell_eval2_runtime_version": "0.0.0+unknown",
        "cell_eval2_module": "/fixture/cell_eval2/__init__.py",
        "pdex_version": "0.3.0",
    }


def write_run_meta(path: Path, truth_view: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "config_digest": CONFIG_DIGEST,
                "cell_eval2_version": "0.0.0+unknown",
                "resolved_device": "cpu",
                "resolved_de_backend": "pdex",
                "comparator": "bulk_lognorm",
                "source_fingerprint_strict": True,
                "source_fingerprint": SOURCE_FINGERPRINT,
                "anchor_semantic_identity": ANCHOR_IDENTITY,
                "input_type_real_effective": "counts",
                "input_type_pred_effective": "counts",
                "source": str(truth_view.resolve()),
                "environment": {"pdex": {"version": "0.3.0"}},
            }
        ),
        encoding="utf-8",
    )


def scorer_outputs(results: Path, *, mtime: bool) -> dict[str, Any]:
    root = results.parent
    return {
        "results": descriptor(results, mtime=mtime),
        "aggregate_results": descriptor(root / "agg_results.csv", mtime=mtime),
        "metric_aggregation": descriptor(
            root / "metric_aggregation.csv", mtime=mtime
        ),
        "run_meta": descriptor(root / "run_meta.json", mtime=mtime),
    }


def write_v51_receipt(
    path: Path,
    *,
    context: str,
    anchor_results: Path,
    candidate_results: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = {
        "config_digest": CONFIG_DIGEST,
        "resolved_device": "cpu",
        "resolved_de_backend": "pdex",
        "cell_eval2_version": "0.0.0+unknown",
        "pdex_version": "0.3.0",
    }
    outputs: dict[str, Any] = {}
    predictions: dict[str, dict[str, Any]] = {}
    initialized_result_dirs: set[Path] = set()
    for role, results in (
        ("anchor", anchor_results),
        ("v51_alpha010", candidate_results),
    ):
        source_prediction = path.parent / f"{role}_source_prediction.h5ad"
        prediction_view = path.parent / f"{role}_prediction_view.h5ad"
        truth_view = path.parent / f"{role}_truth_view.h5ad"
        source_prediction.write_bytes(f"{role}-source".encode("utf-8"))
        prediction_view.write_bytes(f"{role}-prediction".encode("utf-8"))
        truth_view.write_bytes(b"truth")
        result_dir = results.parent.resolve()
        if result_dir not in initialized_result_dirs:
            write_run_meta(results.parent / "run_meta.json", truth_view)
            initialized_result_dirs.add(result_dir)
        views_report = path.parent / f"{role}_views.json"
        views_report.write_text(
            json.dumps(
                {
                    "schema": "vcc-public-scoring-views-v1",
                    "context": context,
                    "axes": {},
                    "contract": {
                        "model_read_treated_truth": False,
                        "input_type": "raw-nonnegative-integer-counts",
                    },
                    "provenance": {
                        "common_genes": {},
                        "common_genes_json": {},
                        "controls_only": {},
                        "output_prediction": descriptor(
                            prediction_view, mtime=False
                        ),
                        "output_truth": descriptor(truth_view, mtime=False),
                        "prediction": descriptor(source_prediction),
                        "sealed_truth": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        outputs[role] = {
            **scorer_outputs(results, mtime=False),
            "views_report": descriptor(views_report, mtime=False),
        }
        source_record = descriptor(source_prediction, mtime=False)
        predictions[role] = {
            "path": source_record["path"],
            "sha256": source_record["sha256"],
        }
    receipt = {
        "schema": "vcc-public-validation-scoring-receipt-v1",
        "status": "scoring_complete",
        "context": context,
        "full_generation_summary": {},
        "predictions": predictions,
        "generation_qc": {},
        "scoring_software": software(),
        "resolved_scoring": {
            "anchor": dict(resolved),
            "v51_alpha010": dict(resolved),
        },
        "outputs": outputs,
    }
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def write_v6_receipt(
    receipt_path: Path,
    *,
    context: str,
    label: str,
    results: Path,
    expected_targets: int,
    target_remaining_fraction: float,
    residual_alpha: float,
    manifest: Path,
) -> tuple[Path, Path]:
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_root = receipt_path.parent / "candidate_contract"
    candidate_root.mkdir(parents=True, exist_ok=True)
    spec = candidate_root / "spec.json"
    prediction = candidate_root / "prediction.h5ad"
    generation_verification = candidate_root / "generation_verified.json"
    prediction.write_bytes(label.encode("utf-8"))
    configuration = {
        "context": context,
        "target_remaining_fraction": target_remaining_fraction,
        "residual_alpha": residual_alpha,
        "seed": 20260901,
    }
    shared_input = descriptor(manifest)
    inputs = {
        name: dict(shared_input)
        for name in (
            "controls_h5ad",
            "panel_manifest",
            "panel_csv",
            "residual_npz",
            "residual_json",
            "checkpoint",
            "selection_json",
            "support_genes",
            "generator_script",
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
            "perturbation_map",
            "var_dims",
        )
    }
    inputs["state_config"] = None
    spec.write_text(
        json.dumps(
            {
                "schema": "vcc-public-candidate-spec-v2",
                "context": context,
                "output_tag": label,
                "configuration": configuration,
                "inputs": inputs,
                "state_source": {
                    "path": str((receipt_path.parent / "state_source").resolve()),
                    "repository_commit": "1" * 40,
                    "repository_tree": "2" * 40,
                    "tracked_worktree_clean": True,
                    "untracked_and_ignored_runtime_files_absent": True,
                    "runtime_isolation": {
                        "mode": "authenticated-clean-worktree-no-bytecode-v1",
                        "pythonpath_first": str((receipt_path.parent / "state_source/src").resolve()),
                        "python_dont_write_bytecode": True,
                    },
                },
                "provenance_contract": {
                    "schema": "vcc-public-generator-provenance-v2",
                    "mode": "strict",
                    "generator_report_provenance_authenticated": True,
                    "optional_state_config_bound": True,
                },
            }
        ),
        encoding="utf-8",
    )
    generation_verification.write_text(
        json.dumps(
            {
                "schema": "vcc-public-candidate-verification-v1",
                "status": "passed",
                "context": context,
                "output_tag": label,
                "configuration": configuration,
                "checks": {
                    "structural": True,
                    "strict_generator_provenance_authenticated": True,
                },
                "provenance": {
                    "spec": descriptor(spec),
                    "prediction_h5ad": descriptor(prediction),
                },
            }
        ),
        encoding="utf-8",
    )
    prediction_view = receipt_path.parent / "prediction_view.h5ad"
    truth_view = receipt_path.parent / "truth_view.h5ad"
    prediction_view.write_bytes(b"prediction-view")
    truth_view.write_bytes(b"truth")
    write_run_meta(results.parent / "run_meta.json", truth_view)
    views = receipt_path.parent / "views.json"
    views.write_text(
        json.dumps(
            {
                "schema": "vcc-public-scoring-views-v1",
                "context": context,
                "axes": {},
                "contract": {},
                "provenance": {
                    "common_genes": {},
                    "common_genes_json": {},
                    "controls_only": {},
                    "output_prediction": descriptor(prediction_view, mtime=False),
                    "output_truth": descriptor(truth_view, mtime=False),
                    "prediction": descriptor(prediction),
                    "sealed_truth": {},
                },
            }
        ),
        encoding="utf-8",
    )
    outputs = scorer_outputs(results, mtime=True)
    receipt = {
        "schema": "vcc-public-cell-eval-run-verification-v1",
        "status": "passed",
        "context": context,
        "output_tag": label,
        "software": software(),
        "resolved_run": {
            "config_digest": CONFIG_DIGEST,
            "resolved_device": "cpu",
            "resolved_de_backend": "pdex",
            "comparator": "bulk_lognorm",
            "source_fingerprint_strict": True,
            "pdex_version": "0.3.0",
        },
        "contract": {
            "aggregate_mse_matches_cell_eval2": True,
            "derived_mse_reconstructed_from_complete_components": True,
            "direction_nan_targets_are_retained_and_counted": True,
            "metric_aggregation_n_used_authenticated": True,
            "nmae_real_side_omission_is_explicit": True,
        },
        "metrics": {},
        "candidate_receipts": {
            "spec": descriptor(spec),
            "generation_verification": descriptor(generation_verification),
            "views": descriptor(views),
            "prediction_view": descriptor(prediction_view),
            "truth_view": descriptor(truth_view),
        },
        "outputs": {
            **outputs,
            "expected_targets": expected_targets,
            "validated_all_context_means": {},
        },
    }
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, spec
