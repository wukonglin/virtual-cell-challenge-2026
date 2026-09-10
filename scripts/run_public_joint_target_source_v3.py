"""Freeze and run the fixed public-only joint target/source engineering screen."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

import numpy as np
import prepare_public_joint_target_source_v3 as preparation
import public_joint_target_source_v3 as model
from public_joint_target_source_tracking import JointTargetSourceTracker
from prepare_public_control_anchor_v2 import load_cache
from train_public_flow_pilot import authenticated_bytes, authenticated_json, load_npz, sha256_file, write_new

SCHEMA = "public-joint-target-source-execution-v3"
MODULES = ("run_public_joint_target_source_v3.py", "prepare_public_joint_target_source_v3.py",
    "public_joint_target_source_v3.py", "public_joint_target_source_tracking.py", "wandb_training.py",
    "public_control_anchor_v2.py", "public_mean_effect_baseline.py", "public_flow_stable_conditioning.py",
    "prepare_public_control_anchor_v2.py", "prepare_public_flow_pilot.py", "train_public_flow_pilot.py",
    "public_flow_conditioning.py", "public_crispri_flow.py", "build_go_target_features.py")
SOURCES = {"nadig_jurkat", "replogle_k562"}
CORE_METRICS = ("target_pooled_mse", "control_target_pooled_mse", "raw_target_pooled_mse",
    "model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
    "negative_fraction", "zero_fraction", "control_zero_fraction", "projection_fraction",
    "projection_mean_absolute_correction", "projection_centroid_distortion_rms")
PREDICTION_KEYS = ("raw", "nonnegative", "delta", "group", "control_cache_rows", "treated_cache_rows",
    "treated_group", "held_groups", "held_source", "group_control_mean", "group_treated_mean",
    "group_raw_mean", "group_predicted_mean")
WEIGHT_KEYS = ("coefficients", "context_mean", "context_scale", "fit_targets", "held_targets", "fit_source",
    "held_source", "aligned_target_ids", "aligned_values", "aligned_present", "feature_ids",
    "feature_signature_json", "feature_receipt_json")
TRANSFORM_KEYS = ("target_mean", "target_scale", "target_transform_provenance_json")
# Independent, already reported model-free reliability values on exactly these cells.
CONTROL_POOLED = {"nadig_jurkat": 0.010447375812918125, "replogle_k562": 0.007270140964519972}


def dump_new(path, value):
    write_new(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def screen_policy():
    return {"primary": "target_pooled_mse", "minimum_relative_improvement": 0.01,
        "sources": sorted(SOURCES), "all_shuffle_seeds": list(model.SHUFFLE_SEEDS),
        "control_and_each_shuffle": "at_least_1_percent_improvement_in_each_source",
        "context": "strict_primary_improvement_in_each_source",
        "secondary_mmd": "noninferior_to_every_comparator_in_each_source",
        "finite_nonnegative_and_identity_required": True, "automatic_promotion": False}


def register(role_path, role_sha, output_dir):
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh execution registration required")
    record = preparation.load_registration(role_path, role_sha)  # Opaque expression bytes only.
    output_dir.mkdir(mode=0o700)
    contract = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "registration": {"path": str(role_path.resolve()), "sha256": role_sha},
        "inputs": record["inputs"], "policy": model.policy(), "screen": screen_policy(),
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "expression_decoded": False, "new_raw_expression_allowed": False, "automatic_promotion": False,
        "stage": "validation", "cpu_only": True, "wandb_mode": "offline", "events_per_arm": 9}
    dump_new(output_dir / "contract.json", contract)
    return sha256_file(output_dir / "contract.json")


def validate_contract(path, digest):
    value = authenticated_json(path, digest)
    if (value.get("schema") != SCHEMA or value.get("policy") != model.policy()
        or value.get("screen") != screen_policy() or value.get("expression_decoded") is not False
        or value.get("new_raw_expression_allowed") is not False or value.get("automatic_promotion") is not False
        or value.get("stage") != "validation" or value.get("cpu_only") is not True
        or value.get("wandb_mode") != "offline" or value.get("events_per_arm") != 9):
        raise ValueError("Registered execution policy changed")
    codes = value.get("code_sha256", {})
    if set(codes) != set(MODULES) or any(sha256_file(Path(__file__).with_name(k)) != v for k, v in codes.items()):
        raise ValueError("Implementation changed after registration")
    binding = value["registration"]
    record = preparation.load_registration(Path(binding["path"]), binding["sha256"])
    if value.get("inputs") != record["inputs"]:
        raise ValueError("Input bindings differ from authenticated role registration")
    return value, record


def verify_metrics(metrics):
    if not isinstance(metrics, dict) or not set(CORE_METRICS) <= set(metrics):
        raise ValueError("Missing registered metric")
    for key in CORE_METRICS:
        value = metrics[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Nonfinite or invalid nonnegative metric")
        if key.endswith("fraction") and value > 1:
            raise ValueError("Invalid fraction")
    if metrics["negative_fraction"] != 0:
        raise ValueError("Negative final prediction")


def verify_summary(summary, arm, contract_sha=None):
    if (summary.get("schema") != "public-joint-target-source-summary-v3" or summary.get("arm") != arm
        or summary.get("policy") != model.policy() or summary.get("stage") != "validation"
        or summary.get("status") != "completed_unpromoted_diagnostic"
        or summary.get("prior_public_development_exposure_acknowledged") is not True):
        raise ValueError("Arm summary policy mismatch")
    if contract_sha is not None and (summary.get("diagnostic_contract_sha256") != contract_sha
        or summary.get("provenance", {}).get("diagnostic_contract_sha256") != contract_sha):
        raise ValueError("Arm contract provenance mismatch")
    for key in ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified"):
        if summary.get(key) is not True:
            raise ValueError("Required anchor identity failed")
    for key in ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
                "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
                "held_roles_used_for_fitting"):
        if summary.get(key) is not False:
            raise ValueError("Forbidden fitting, promotion or data use")
    verify_metrics(summary.get("aggregate"))
    rows = summary.get("source_metrics", [])
    if len(rows) != 2 or {r["source"] for r in rows} != SOURCES:
        raise ValueError("Both public sources required")
    for row in rows:
        verify_metrics(row["metrics"])
    folds = summary.get("folds", [])
    if len(folds) != 8 or len({f["fold_id"] for f in folds}) != 8:
        raise ValueError("Exactly eight distinct outer directions required")
    for fold in folds:
        verify_metrics(fold["metrics"])
        if len(fold["fit_targets"]) != 42 or len(fold["held_targets"]) != 14:
            raise ValueError("42/14 target split changed")


def check_replayed_metrics(reported, vectors, groups):
    """Independently pool error vectors; reaggregate population diagnostics."""
    computed = model.aggregate_group_metrics(groups)
    for name, prediction_key in (("target_pooled_mse", "predicted_mean"),
        ("raw_target_pooled_mse", "raw_mean"), ("control_target_pooled_mse", "control_mean")):
        source_values = []
        for source in sorted({r["source"] for r in vectors}):
            target_values = []
            for target in sorted({r["target"] for r in vectors if r["source"] == source}):
                selected = [r for r in vectors if (r["source"], r["target"]) == (source, target)]
                error = np.mean([r[prediction_key] - r["treated_mean"] for r in selected], axis=0)
                target_values.append(float(np.mean(error ** 2)))
            source_values.append(float(np.mean(target_values)))
        computed[name] = float(np.mean(source_values))
    for key in CORE_METRICS:
        if not math.isclose(computed[key], reported[key], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError(f"Saved metric replay differs: {key}")


def verify_saved_predictions(output_dir, summary, arrays, registration):
    """Independently replay anchors/projection/means and role-specific cache rows."""
    folds = model.joint_folds(tuple(arrays["targets"].astype(str)), tuple(arrays["sources"].astype(str)), registration)
    prefixes = [f"fold_{i}_" for i in range(8)]
    predictions = load_npz(output_dir / "predictions.npz", summary["predictions_sha256"],
        {"genes"} | {p + key for p in prefixes for key in PREDICTION_KEYS})
    weight_names = WEIGHT_KEYS + (() if summary["arm"] in ("control", "context") else TRANSFORM_KEYS)
    weights = load_npz(output_dir / "weights.npz", summary["weights_sha256"],
        {"genes"} | {p + key for p in prefixes for key in weight_names})
    if not np.array_equal(predictions["genes"], arrays["genes"]) or not np.array_equal(weights["genes"], arrays["genes"]):
        raise ValueError("Output gene axis changed")
    coverage, vectors, diagnostic_groups = set(), [], []
    for fold, prefix, report in zip(folds, prefixes, summary["folds"]):
        if report["fold_id"] != fold.fold_id or report["array_prefix"] != prefix:
            raise ValueError("Fold serialization changed")
        if (tuple(weights[prefix + "fit_targets"].astype(str)) != tuple(sorted(fold.fit_targets))
            or tuple(weights[prefix + "held_targets"].astype(str)) != fold.held_targets
            or str(weights[prefix + "fit_source"].item()) != fold.fit_source
            or str(weights[prefix + "held_source"].item()) != fold.held_source):
            raise ValueError("Saved fit/held roles changed")
        groups = predictions[prefix + "held_groups"]
        if not np.array_equal(groups, fold.evaluation_indices) or str(predictions[prefix + "held_source"].item()) != fold.held_source:
            raise ValueError("Saved evaluation group roles changed")
        delta = predictions[prefix + "delta"]
        if delta.shape != (len(groups), len(arrays["genes"])) or not np.isfinite(delta).all():
            raise ValueError("Invalid saved effect shape/values")
        target_ids = tuple(weights[prefix + "aligned_target_ids"].astype(str))
        if target_ids != tuple(sorted(set(fold.fit_targets) | set(fold.held_targets))):
            raise ValueError("Saved feature target axis differs")
        context = (arrays["context"][groups] - weights[prefix + "context_mean"]) / weights[prefix + "context_scale"]
        parts = [np.ones((len(groups), 1))]
        if summary["arm"] not in ("control", "context"):
            features = (weights[prefix + "aligned_values"] - weights[prefix + "target_mean"]) / weights[prefix + "target_scale"]
            present = weights[prefix + "aligned_present"]
            if present.dtype != np.bool_ or present.shape != (len(target_ids), 1):
                raise ValueError("Saved GO modality mask differs")
            features[~present[:, 0]] = 0
            features = np.column_stack((features, present, ~present.any(1)))
            rows = [target_ids.index(str(arrays["targets"][g])) for g in groups]
            parts.append(features[rows] / math.sqrt(features.shape[1]))
        parts.append(context / math.sqrt(context.shape[1]))
        replayed_delta = np.column_stack(parts) @ weights[prefix + "coefficients"]
        if not np.allclose(replayed_delta, delta, rtol=1e-12, atol=1e-14):
            raise ValueError("Saved weights/transforms do not replay predicted effect")
        ci = np.concatenate([np.flatnonzero(arrays["control_group"] == g) for g in groups])
        ti = np.concatenate([np.flatnonzero(arrays["group"] == g) for g in groups])
        pg = np.concatenate([np.full(np.sum(arrays["control_group"] == g), g, dtype=np.int64) for g in groups])
        tg = np.concatenate([np.full(np.sum(arrays["group"] == g), g, dtype=np.int64) for g in groups])
        for key, expected in (("control_cache_rows", ci), ("treated_cache_rows", ti), ("group", pg), ("treated_group", tg)):
            if not np.array_equal(predictions[prefix + key], expected):
                raise ValueError("Saved prediction-to-cache row mapping differs")
        raw, positive = predictions[prefix + "raw"], predictions[prefix + "nonnegative"]
        lookup = {int(g): i for i, g in enumerate(groups)}
        expected_raw = arrays["control"][ci].astype(np.float64) + delta[[lookup[int(g)] for g in pg]]
        if (not np.isfinite(raw).all() or not np.isfinite(positive).all()
            or not np.array_equal(raw, expected_raw) or not np.array_equal(positive, np.maximum(expected_raw, 0))):
            raise ValueError("Saved output fails exact control-anchored projection replay")
        if summary["arm"] == "control" and not np.array_equal(positive, arrays["control"][ci]):
            raise ValueError("Control arm changed expression")
        fold_vectors, fold_diagnostics = [], []
        for position, group in enumerate(groups):
            target, source, batch = (str(arrays[key][group]) for key in ("targets", "sources", "batches"))
            identity = (source, target, batch)
            if identity in coverage:
                raise ValueError("Repeated evaluation group")
            coverage.add(identity)
            means = {"control_mean": arrays["control"][ci[pg == group]].astype(np.float64).mean(0),
                "treated_mean": arrays["treated"][ti[tg == group]].astype(np.float64).mean(0),
                "raw_mean": raw[pg == group].mean(0), "predicted_mean": positive[pg == group].mean(0)}
            for key, expected in means.items():
                if not np.array_equal(predictions[prefix + "group_" + key][position], expected):
                    raise ValueError("Saved group mean differs from mapped cells")
            vector = {"source": source, "target": target, "batch": batch, **means}
            vectors.append(vector); fold_vectors.append(vector)
            control = arrays["control"][ci[pg == group]]
            treated = arrays["treated"][ti[tg == group]]
            anchored = model.anchor_expression(control, delta[position], is_ntc=np.zeros(len(control), dtype=bool),
                representation=model.REPRESENTATION)
            paired = model.paired_population_diagnostics(anchored, control, treated, representation=model.REPRESENTATION)
            diagnostics = {"source": source, "target": target, "batch": batch,
                **paired["nonnegative"], **paired["projection"], "control_zero_fraction": float(np.mean(control == 0))}
            diagnostic_groups.append(diagnostics); fold_diagnostics.append(diagnostics)
        check_replayed_metrics(report["metrics"], fold_vectors, fold_diagnostics)
    if len(coverage) != 398:
        raise ValueError("All 398 fixed groups must occur out of fold once")
    for source in sorted(SOURCES):
        reported = next(r["metrics"] for r in summary["source_metrics"] if r["source"] == source)
        check_replayed_metrics(reported, [r for r in vectors if r["source"] == source],
            [r for r in diagnostic_groups if r["source"] == source])
        if not math.isclose(reported["control_target_pooled_mse"], CONTROL_POOLED[source], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("Independent reliability/control baseline identity differs")
    check_replayed_metrics(summary["aggregate"], vectors, diagnostic_groups)
    return {"exact_projection_replay": True, "cache_row_mapping_verified": True,
        "saved_weights_and_transforms_replayed": True, "secondary_mmd_replayed": True,
        "fold_source_global_core_metrics_replayed": True, "target_pooled_mse_replayed": True,
        "out_of_fold_groups": len(coverage), "out_of_fold_source_targets": 112}


def screen(summaries):
    if set(summaries) != set(model.ARMS):
        raise ValueError("All six predeclared arms required")
    for arm, summary in summaries.items():
        verify_summary(summary, arm)
    results = []
    for source in sorted(SOURCES):
        values = {arm: next(r["metrics"] for r in s["source_metrics"] if r["source"] == source)
                  for arm, s in summaries.items()}
        true = values["true"]
        gains = {arm: 1.0 - true["target_pooled_mse"] / v["target_pooled_mse"]
            if v["target_pooled_mse"] > 0 else None for arm, v in values.items() if arm != "true"}
        checks = {f"primary_vs_{arm}": gains[arm] is not None and gains[arm] >= 0.01
            for arm in ("control", *(f"shuffled_{s}" for s in model.SHUFFLE_SEEDS))}
        checks["primary_vs_context"] = true["target_pooled_mse"] < values["context"]["target_pooled_mse"]
        for arm, value in values.items():
            if arm != "true":
                checks[f"mmd_vs_{arm}"] = true["model_mmd2"] <= value["model_mmd2"]
        results.append({"source": source, "relative_primary_improvement": gains, "checks": checks,
                        "passed": all(checks.values())})
    return {"policy": screen_policy(), "sources": results, "passed": all(r["passed"] for r in results),
        "automatic_promotion": False, "leaderboard_improvement_claimed": False}


def run(contract_path, contract_sha, output_dir):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Requires an approved BioHPC compute host, never a login/head node")
    contract, registration = validate_contract(contract_path, contract_sha)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh suite output required")
    output_dir.mkdir(mode=0o700)
    inputs, paths = registration["inputs"], {k: Path(v) for k, v in registration["input_paths"].items()}
    tables = preparation.load_feature_tables(registration)
    arrays, row_contract = load_cache(paths["cache"], inputs["cache_sha256"], paths["contract"], inputs["contract_sha256"],
        paths["source_manifest"], inputs["source_manifest_sha256"])
    if row_contract["excluded_targets"] != registration["excluded_targets"]:
        raise ValueError("Excluded target roster changed")
    for value in arrays.values():
        value.setflags(write=False)
    summaries, runs = {}, []
    for arm in model.ARMS:
        tracker = JointTargetSourceTracker(output_dir / "tracking" / arm, arm=arm, config={**inputs,
            "diagnostic_contract_sha256": contract_sha, "split_sha256": contract["registration"]["sha256"],
            "entrypoint_sha256": contract["code_sha256"]["run_public_joint_target_source_v3.py"]})
        exit_code = 1
        arm_dir = output_dir / "runs" / arm
        arm_dir.parent.mkdir(exist_ok=True)
        try:
            def progress(event):
                tracker.log_joint(event["held_source"], event["metrics"])
                print(json.dumps({"arm": arm, "completed_fold": event["fold_index"] + 1,
                    "held_source": event["held_source"], "target_pooled_mse": event["metrics"]["target_pooled_mse"]}), flush=True)
            summary = model.run_arm(arrays, registration, tables, arm, arm_dir,
                {"diagnostic_contract_sha256": contract_sha, "role_manifest_sha256": contract["registration"]["sha256"],
                 "inputs": inputs}, progress=progress)
            verify_summary(summary, arm, contract_sha)
            audit = verify_saved_predictions(arm_dir, summary, arrays, registration)
            tracker.log_joint("all", summary["aggregate"])
            exit_code = 0
        finally:
            tracker.finish(exit_code)
        tracking_path = output_dir / "tracking" / arm / "tracking.json"
        receipt = authenticated_json(tracking_path, sha256_file(tracking_path))
        if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("event_count") != 9 or receipt.get("tracking_errors") != 0
            or receipt.get("mode") != "offline" or receipt.get("stage") != "validation"
            or receipt.get("cloud_synced") is not False):
            raise ValueError("Offline W&B recording incomplete")
        summaries[arm] = summary
        runs.append({"arm": arm, "summary_sha256": sha256_file(arm_dir / "summary.json"),
            "tracking_sha256": sha256_file(tracking_path), "run_id": receipt["run_id"],
            "tracking_events": 9, "tracking_errors": 0, "cloud_synced": False, "artifact_audit": audit})
    if len({r["run_id"] for r in runs}) != 6:
        raise ValueError("Distinct W&B run identities required")
    validate_contract(contract_path, contract_sha)  # Recheck implementation/old seals after execution.
    suite = {"schema": SCHEMA, "completed": True, "diagnostic_contract_sha256": contract_sha,
        "decision": screen(summaries), "runs": runs, "source_metrics": {a: s["source_metrics"] for a, s in summaries.items()},
        "training_performed": "40 fixed diagnostic ridge fits", "pretraining_performed": False,
        "posttraining_performed": False, "promoted": False, "submission_performed": False,
        "new_raw_expression_read": False, "download_performed": False, "gpu_used": False,
        "cloud_synced": False, "tracking_events": 54, "tracking_errors": 0,
        "runtime": {"hostname": socket.gethostname(), "device": "cpu", "numpy_version": np.__version__}}
    dump_new(output_dir / "suite.json", suite)
    completion = {"schema": SCHEMA, "completed": True, "suite_sha256": sha256_file(output_dir / "suite.json"),
        "screen_passed": suite["decision"]["passed"], "tracking_events": 54, "tracking_errors": 0,
        "cloud_synced": False, "promoted": False, "submission_performed": False}
    dump_new(output_dir / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--role-manifest-sha256")
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "register":
        if args.role_manifest is None or not args.role_manifest_sha256:
            parser.error("Registration requires role manifest and SHA")
        print(json.dumps({"contract_sha256": register(args.role_manifest, args.role_manifest_sha256, args.output_dir)}))
        return 0
    if args.contract is None or not args.contract_sha256:
        parser.error("Execution requires frozen contract and SHA")
    return run(args.contract, args.contract_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
