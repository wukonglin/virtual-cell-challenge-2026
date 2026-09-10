"""Public target-by-context experiment with the frozen v4 sparse output.

Only authenticated existing public cache arrays are admitted by the caller.
This is a small diagnostic, not a flow parent, count emitter or submission.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import socket
import numpy as np

import public_joint_target_source_v3 as v3
import public_sparse_hurdle_v4 as sparse
import public_target_context_kernel_v5 as kernel
from public_mean_effect_baseline import balanced_weights, save_arrays
from train_public_flow_pilot import load_npz, write_new

ARMS = ("control", "context", "additive", "true", "shuffled_20260921", "shuffled_20260922", "shuffled_20260923")
SHUFFLED_ARMS = ARMS[4:]
COMMON_KEYS = ("fit_targets", "held_targets", "fit_source", "held_source",
    "aligned_target_ids", "aligned_values", "aligned_present", "feature_ids",
    "feature_signature_json", "feature_receipt_json", "positive_prior", "prior_positive_counts",
    "training_positive_support", "prior_source_ids", "prior_original_rows", "prior_pool_kind",
    "training_group_ids", "training_response")


def policy():
    return {"schema": "public-target-context-model-v5", "arms": list(ARMS),
        "kernel": kernel.policy(), "decoder": sparse.policy(),
        "training_diagnostics": "in-sample descriptive only; no model or hyperparameter selection",
        "comparison": "shared nonlinear context basis, not matched effective capacity",
        "cross_source_limit": "one fitting source per direction; within-source interactions extrapolate to held source",
        "hyperparameter_selection": False, "posttraining": False, "count_emitter": False,
        "prior_public_development_exposure_acknowledged": True}


def feature_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown interaction arm")
    return "true" if arm == "additive" else arm


def fit_arguments(arrays, fold):
    return dict(training_targets=tuple(str(arrays["targets"][i]) for i in fold.fit_indices),
        training_sources=tuple(str(arrays["sources"][i]) for i in fold.fit_indices),
        context=arrays["context"][list(fold.fit_indices)])


def training_diagnostics(decoder, response, arrays, fold, aligned):
    result = dict(kernel.training_diagnostics(decoder, response))
    targets = tuple(str(arrays["targets"][i]) for i in fold.fit_indices)
    weights = balanced_weights(targets, tuple(str(arrays["sources"][i]) for i in fold.fit_indices))
    heads = []
    for group in fold.fit_indices:
        packet = v3._subset(aligned, (str(arrays["targets"][group]),))
        heads.append(decoder.predict_delta(packet, arrays["context"][group:group+1],
            is_ntc=np.array([False]))[0])
    heads = np.asarray(heads)
    width = arrays["control"].shape[1]
    unchanged = []
    for head, group in zip(heads, fold.fit_indices):
        reference = arrays["control"][arrays["control_group"] == group]
        n = len(reference)
        count = np.sum(reference > 0, axis=0)
        q = count / n
        smoothed = (32*q+.5)/33
        delta = np.clip(head[:width], -sparse.OCCUPANCY_CLIP, sparse.OCCUPANCY_CLIP)
        shifted = 1 / (1 + np.exp(-(np.log(smoothed)-np.log1p(-smoothed)+delta)))
        requested = np.clip(q+shifted-smoothed, 0, 1)
        requested[delta == 0] = q[delta == 0]
        desired = np.floor(n*requested+.5).astype(np.int64)
        unchanged.append(float(np.mean(desired == count)))
    result["training_unchanged_count_fraction"] = float(weights @ unchanged)
    if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError("Invalid training-only diagnostic")
    return result


def model_arrays(decoder, prior, response, fold, aligned, receipt, table):
    result = {"fit_targets": np.asarray(fold.fit_targets), "held_targets": np.asarray(fold.held_targets),
        "fit_source": np.asarray(fold.fit_source), "held_source": np.asarray(fold.held_source),
        "aligned_target_ids": np.asarray(aligned.target_ids), "aligned_values": aligned.values,
        "aligned_present": aligned.present, "feature_ids": np.asarray(table.feature_ids),
        "feature_signature_json": np.asarray(json.dumps(aligned.signature)),
        "feature_receipt_json": np.asarray(json.dumps(receipt, sort_keys=True)),
        "positive_prior": prior.values, "prior_positive_counts": prior.positive_counts,
        "training_positive_support": prior.supported, "prior_source_ids": prior.source_ids,
        "prior_original_rows": prior.original_rows, "prior_pool_kind": prior.pool_kind,
        "training_group_ids": np.asarray(fold.fit_indices, dtype=np.int64), "training_response": response}
    result.update({"kernel_"+k: value for k, value in kernel.decoder_arrays(decoder).items()})
    return result


def finish_report(report, number, fold, receipt, decoder, prior, train_report):
    report.update(array_prefix=f"fold_{number}_", feature_receipt=receipt,
        target_transform=decoder.target_transform.provenance() if decoder.target_transform else None,
        prior_unique_training_rows=len(prior.original_rows),
        prior_unique_anchor_rows=int(np.sum(prior.pool_kind == 1)),
        prior_unique_treated_rows=int(np.sum(prior.pool_kind == 2)),
        unsupported_genes=int(np.sum(~prior.supported)), training_diagnostics=train_report)


def run_arm(arrays, registration, tables, arm, output_dir, provenance, progress=None):
    if arm not in ARMS:
        raise ValueError("Unknown interaction arm")
    folds = v3.joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    if set(tables) != {f.target_fold for f in folds}:
        raise ValueError("Authenticated fold-local GO tables required")
    if not isinstance(provenance.get("diagnostic_contract_sha256"), str) or len(provenance["diagnostic_contract_sha256"]) != 64:
        raise ValueError("Frozen diagnostic contract required")
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh interaction output directory required")
    output_dir.mkdir(mode=0o700)
    reports, weights, predictions, events = [], {}, {}, []
    for number, fold in enumerate(folds):
        aligned, receipt = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets,
            arm=feature_arm(arm), fold_id=fold.target_fold)
        receipt = {**receipt, "consumer_arm": arm}
        prior = sparse.training_positive_prior(arrays, fold)
        response = sparse.training_responses(arrays, fold, prior)
        args = fit_arguments(arrays, fold)
        decoder = kernel.fit_kernel_decoder(args["training_targets"], args["training_sources"], args["context"],
            response, aligned=aligned, arm=arm, fold=fold, representation=sparse.REPRESENTATION)
        train_report = training_diagnostics(decoder, response, arrays, fold, aligned)
        report, output = sparse.evaluate_fold(arrays, fold, aligned, decoder, prior)
        finish_report(report, number, fold, receipt, decoder, prior, train_report)
        prefix = f"fold_{number}_"
        weights.update({prefix+k: value for k, value in model_arrays(decoder, prior, response, fold,
            aligned, receipt, tables[fold.target_fold]).items()})
        predictions.update({prefix+k: value for k, value in output.items()})
        reports.append(report)
        event = {"fold_index": number, "held_source": fold.held_source, "target_fold": fold.target_fold,
            "metrics": report["metrics"], "training_diagnostics": train_report}
        events.append(event)
        if progress is not None:
            progress(event)
    sources, aggregate = sparse._final_metrics(reports)
    weights["genes"] = predictions["genes"] = arrays["genes"]
    summary = {"schema": "public-target-context-summary-v5", "status": "completed_unpromoted_diagnostic",
        "stage": "validation", "arm": arm, "policy": policy(), "folds": reports,
        "source_metrics": sources, "aggregate": aggregate, "provenance": provenance,
        "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "weights_sha256": save_arrays(output_dir / "weights.npz", weights),
        "predictions_sha256": save_arrays(output_dir / "predictions.npz", predictions),
        "exact_ntc_identity_verified": all(r["exact_ntc_identity_verified"] for r in reports),
        "exact_zero_effect_identity_verified": all(r["exact_zero_effect_identity_verified"] for r in reports),
        "count_emitter": False, "posttraining_performed": False, "promoted": False, "submission_performed": False,
        "hyperparameter_selection": False, "H1_treated_read": False, "RPE1_treated_read": False,
        "challenge_treated_used": False, "held_roles_used_for_fitting": False,
        "prior_public_development_exposure_acknowledged": True, "learned_null_calibration_evaluated": False,
        "runtime": {"hostname": socket.gethostname(), "device": "cpu", "numpy_version": np.__version__}}
    write_new(output_dir / "steps.jsonl", b"".join((json.dumps(e, sort_keys=True, allow_nan=False)+"\n").encode() for e in events))
    write_new(output_dir / "summary.json", (json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+"\n").encode())
    return summary


def verify_artifacts(output_dir, summary, arrays, registration, tables):
    """Reconstruct training transforms and replay saved predictions; never refit."""
    summary = json.loads(json.dumps(summary, allow_nan=False))
    arm = summary["arm"]
    if (arm not in ARMS or summary.get("schema") != "public-target-context-summary-v5"
        or summary.get("status") != "completed_unpromoted_diagnostic" or summary.get("stage") != "validation"):
        raise ValueError("Invalid interaction summary")
    sparse._same_tree(summary["policy"], json.loads(json.dumps(policy())), "policy")
    for key in ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified", "prior_public_development_exposure_acknowledged"):
        if summary.get(key) is not True:
            raise ValueError("Required identity/exposure flag failed")
    for key in ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
        "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used", "held_roles_used_for_fitting"):
        if summary.get(key) is not False:
            raise ValueError("Forbidden interaction activity")
    folds = v3.joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    if len(summary["folds"]) != len(folds):
        raise ValueError("Fold count mismatch")
    prefixes = [f"fold_{i}_" for i in range(len(folds))]
    state_keys = COMMON_KEYS + tuple("kernel_"+k for k in kernel.decoder_keys(arm))
    weights = load_npz(Path(output_dir)/"weights.npz", summary["weights_sha256"],
        {"genes"} | {p+k for p in prefixes for k in state_keys})
    predictions = load_npz(Path(output_dir)/"predictions.npz", summary["predictions_sha256"],
        {"genes"} | {p+k for p in prefixes for k in sparse.PREDICTION_KEYS})
    if not np.array_equal(weights["genes"], arrays["genes"]) or not np.array_equal(predictions["genes"], arrays["genes"]):
        raise ValueError("Saved gene axis differs")
    replayed = []
    for number, (prefix, fold, saved_report) in enumerate(zip(prefixes, folds, summary["folds"])):
        aligned, receipt = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets,
            arm=feature_arm(arm), fold_id=fold.target_fold)
        receipt = {**receipt, "consumer_arm": arm}
        prior = sparse.training_positive_prior(arrays, fold)
        response = sparse.training_responses(arrays, fold, prior)
        state = {k: weights[prefix+"kernel_"+k] for k in kernel.decoder_keys(arm)}
        decoder = kernel.restore_kernel_decoder(state, arm=arm, fold=fold, aligned=aligned,
            response=response, **fit_arguments(arrays, fold))
        expected_weights = model_arrays(decoder, prior, response, fold, aligned, receipt, tables[fold.target_fold])
        for key, expected in expected_weights.items():
            if not np.array_equal(weights[prefix+key], expected):
                raise ValueError(f"Saved training state replay mismatch: {key}")
        train_report = training_diagnostics(decoder, response, arrays, fold, aligned)
        report, output = sparse.evaluate_fold(arrays, fold, aligned, decoder, prior)
        for key, expected in output.items():
            if not np.array_equal(predictions[prefix+key], expected):
                raise ValueError(f"Saved prediction replay mismatch: {key}")
        finish_report(report, number, fold, receipt, decoder, prior, train_report)
        sparse._same_tree(saved_report, json.loads(json.dumps(report)), "fold")
        replayed.append(report)
    sources, aggregate = sparse._final_metrics(replayed)
    sparse._same_tree(summary["source_metrics"], sources, "source_metrics")
    sparse._same_tree(summary["aggregate"], aggregate, "aggregate")
    return {"passed": True, "ridge_refits": 0, "saved_head_replay": True, "training_prior_replay": True,
        "kernel_system_verified": True, "training_only_transforms_verified": True,
        "deterministic_donor_replay": True, "original_row_mapping_verified": True,
        "pooled_and_batch_metrics_replayed": True, "training_diagnostics_replayed": True,
        "out_of_fold_groups": sum(len(f.evaluation_indices) for f in folds),
        "out_of_fold_source_targets": sum(len(f.held_targets) for f in folds)}

