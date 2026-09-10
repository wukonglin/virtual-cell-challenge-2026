"""Registered GO-by-control-context public validation; no promotion or upload."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

import prepare_public_joint_target_source_v3 as preparation
import public_target_context_v5 as model
import run_public_joint_target_source_v3 as prior
from public_target_context_tracking import METRICS, TargetContextTracker
from prepare_public_control_anchor_v2 import load_cache
from train_public_flow_pilot import authenticated_bytes, authenticated_json, sha256_file, write_new

SCHEMA = "public-target-context-execution-v5"
PRIOR_SHA = "5513a0e751e500d298c4b6e29f5b6205a29160f5444da08a2e4e5a6e2d1cfafb"
SOURCES = {"nadig_jurkat", "replogle_k562"}
MODULES = ("run_public_target_context_v5.py", "public_target_context_v5.py",
           "public_target_context_kernel_v5.py", "public_target_context_tracking.py",
           "public_sparse_hurdle_v4.py", "public_target_context_readiness.py",
           "public_training_readiness.py", *prior.MODULES)
FLAGS_FALSE = ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
    "hyperparameter_selection", "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
    "held_roles_used_for_fitting")


def dump_new(path, value):
    write_new(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def screen_policy():
    return {"minimum_relative_primary_improvement": 0.01,
        "primary_comparators": ["control", *model.SHUFFLED_ARMS],
        "strict_context_primary_improvement": True,
        "strict_additive_primary_improvement": True,
        "batch_centroid_mse_noninferior_to_control_and_context": True,
        "mmd_noninferior_to_all_comparators": True,
        "occupancy_mse_noninferior_to_control_and_context": True,
        "aggregate_zero_rate_error_noninferior_to_control": True,
        "maximum_occupancy_clip_fraction": 0.01, "maximum_amplitude_clip_fraction": 0.01,
        "unresolved_activation_fraction": 0.0,
        "both_sources_required": sorted(SOURCES), "automatic_promotion": False,
        "compatible_parent_and_full_count_emitter_not_established": True}


def register(prior_path, protocol, output_dir):
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh registration required")
    previous, roles = prior.validate_contract(prior_path, PRIOR_SHA)
    output_dir.mkdir(mode=0o700)
    record = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "prior_v3_contract": {"path": str(prior_path.resolve()), "sha256": PRIOR_SHA},
        "protocol": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
        "inputs": roles["inputs"], "registration": previous["registration"],
        "policy": model.policy(), "screen": screen_policy(),
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "expression_decoded": False, "new_raw_expression_allowed": False,
        "automatic_promotion": False, "submission_allowed": False,
        "stage": "validation", "cpu_only": True, "wandb_mode": "offline", "events_per_arm": 9}
    dump_new(output_dir / "contract.json", record)
    return sha256_file(output_dir / "contract.json")


def validate_contract(path, digest):
    value = authenticated_json(path, digest)
    if (value.get("schema") != SCHEMA or value.get("policy") != model.policy()
        or value.get("screen") != screen_policy() or value.get("expression_decoded") is not False
        or value.get("new_raw_expression_allowed") is not False or value.get("automatic_promotion") is not False
        or value.get("submission_allowed") is not False or value.get("stage") != "validation"
        or value.get("cpu_only") is not True or value.get("wandb_mode") != "offline" or value.get("events_per_arm") != 9):
        raise ValueError("Frozen interaction execution policy changed")
    codes = value.get("code_sha256", {})
    if set(codes) != set(MODULES) or any(sha256_file(Path(__file__).with_name(k)) != v for k, v in codes.items()):
        raise ValueError("Interaction runtime implementation changed after registration")
    if value["prior_v3_contract"]["sha256"] != PRIOR_SHA:
        raise ValueError("Only the frozen v3 parent registration is admitted")
    previous, roles = prior.validate_contract(Path(value["prior_v3_contract"]["path"]), PRIOR_SHA)
    if value.get("inputs") != roles["inputs"] or value.get("registration") != previous["registration"]:
        raise ValueError("Input/role binding changed")
    authenticated_bytes(Path(value["protocol"]["path"]), value["protocol"]["sha256"], 1 << 20)
    return value, roles


def verify_metrics(values):
    if not isinstance(values, dict) or not set(METRICS) <= set(values):
        raise ValueError("Missing sparse diagnostic metric")
    for key in METRICS:
        value = values[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Invalid/nonfinite sparse metric")
        if key.endswith("fraction") and value > 1:
            raise ValueError("Invalid sparse fraction")


def verify_summary(summary, arm, contract_sha=None):
    if (summary.get("schema") != "public-target-context-summary-v5"
        or summary.get("status") != "completed_unpromoted_diagnostic"
        or summary.get("arm") != arm or summary.get("stage") != "validation"
        or summary.get("policy") != model.policy()
        or summary.get("prior_public_development_exposure_acknowledged") is not True):
        raise ValueError("Interaction summary policy/status mismatch")
    if contract_sha is not None and (summary.get("diagnostic_contract_sha256") != contract_sha
        or summary.get("provenance", {}).get("diagnostic_contract_sha256") != contract_sha):
        raise ValueError("Interaction summary contract provenance mismatch")
    if any(summary.get(key) is not False for key in FLAGS_FALSE):
        raise ValueError("Forbidden model/data/promotion activity")
    if any(summary.get(key) is not True for key in ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified")):
        raise ValueError("Sparse anchor identity failed")
    verify_metrics(summary.get("aggregate"))
    rows = summary.get("source_metrics", [])
    if len(rows) != 2 or {r["source"] for r in rows} != SOURCES:
        raise ValueError("Exactly both public sources required")
    for row in rows:
        verify_metrics(row["metrics"])
    folds = summary.get("folds", [])
    if len(folds) != 8 or len({f["fold_id"] for f in folds}) != 8:
        raise ValueError("Eight distinct sparse fold directions required")
    for fold in folds:
        verify_metrics(fold["metrics"])
        if len(fold["fit_targets"]) != 42 or len(fold["held_targets"]) != 14:
            raise ValueError("Sparse target role counts differ")


def screen(summaries):
    if set(summaries) != set(model.ARMS):
        raise ValueError("All seven interaction arms required")
    for arm, summary in summaries.items():
        verify_summary(summary, arm)
    rows = []
    for source in sorted(SOURCES):
        values = {arm: next(r["metrics"] for r in s["source_metrics"] if r["source"] == source)
                  for arm, s in summaries.items()}
        true = values["true"]
        gains = {arm: 1 - true["target_pooled_mse"] / v["target_pooled_mse"]
            if v["target_pooled_mse"] > 0 else None for arm, v in values.items() if arm != "true"}
        conditioning = {f"primary_vs_{arm}": values[arm]["target_pooled_mse"] > 0
            and true["target_pooled_mse"] <= (1 - screen_policy()["minimum_relative_primary_improvement"]) * values[arm]["target_pooled_mse"]
            for arm in ("control", *model.SHUFFLED_ARMS)}
        conditioning["primary_vs_context"] = true["target_pooled_mse"] < values["context"]["target_pooled_mse"]
        conditioning["primary_vs_additive"] = true["target_pooled_mse"] < values["additive"]["target_pooled_mse"]
        distribution = {f"mmd_vs_{arm}": true["model_mmd2"] <= v["model_mmd2"]
            for arm, v in values.items() if arm != "true"}
        for arm in ("control", "context"):
            distribution[f"occupancy_vs_{arm}"] = true["target_pooled_occupancy_mse"] <= values[arm]["target_pooled_occupancy_mse"]
            distribution[f"batch_centroid_vs_{arm}"] = true["model_centroid_mse"] <= values[arm]["model_centroid_mse"]
        distribution["aggregate_zero_error_vs_control"] = (
            abs(true["zero_fraction"] - true["treated_zero_fraction"])
            <= abs(true["control_zero_fraction"] - true["treated_zero_fraction"]))
        safety = {"nonnegative": true["negative_fraction"] == 0,
            "no_unresolved_activation": true["unresolved_activation_fraction"] == 0,
            "occupancy_clipping_bounded": true["occupancy_clip_fraction"] <= .01,
            "amplitude_clipping_bounded": true["amplitude_clip_fraction"] <= .01,
            "identities_verified": summaries["true"]["exact_ntc_identity_verified"]
                and summaries["true"]["exact_zero_effect_identity_verified"]}
        checks = {**conditioning, **distribution, **safety}
        rows.append({"source": source, "checks": checks, "relative_primary_improvement": gains,
            "conditioning_passed": all(conditioning.values()), "distribution_passed": all(distribution.values()),
            "safety_passed": all(safety.values()), "passed": all(checks.values())})
    return {"policy": screen_policy(), "sources": rows,
        **{key: all(row[key] for row in rows) for key in ("passed", "conditioning_passed", "distribution_passed", "safety_passed")},
        "automatic_promotion": False, "leaderboard_improvement_claimed": False}


def run(contract_path, contract_sha, output_dir):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Interaction validation requires an approved BioHPC compute host, never a login node")
    contract, roles = validate_contract(contract_path, contract_sha)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh interaction suite output required")
    output_dir.mkdir(mode=0o700)
    paths = {k: Path(v) for k, v in roles["input_paths"].items()}
    inputs = roles["inputs"]
    tables = preparation.load_feature_tables(roles)
    arrays, row_contract = load_cache(paths["cache"], inputs["cache_sha256"], paths["contract"], inputs["contract_sha256"],
        paths["source_manifest"], inputs["source_manifest_sha256"])
    if row_contract["excluded_targets"] != roles["excluded_targets"]:
        raise ValueError("Protected exclusions differ")
    for value in arrays.values():
        value.setflags(write=False)
    summaries, runs = {}, []
    for arm in model.ARMS:
        tracker = TargetContextTracker(output_dir / "tracking" / arm, arm=arm, config={**inputs,
            "diagnostic_contract_sha256": contract_sha, "split_sha256": contract["registration"]["sha256"],
            "entrypoint_sha256": contract["code_sha256"]["run_public_target_context_v5.py"]})
        exit_code = 1
        arm_dir = output_dir / "runs" / arm
        arm_dir.parent.mkdir(exist_ok=True)
        try:
            def progress(event):
                tracker.log_kernel(event["held_source"], event["metrics"], event["training_diagnostics"])
                print(json.dumps({"arm": arm, "completed_fold": event["fold_index"] + 1,
                    "held_source": event["held_source"], "target_pooled_mse": event["metrics"]["target_pooled_mse"],
                    "zero_fraction": event["metrics"]["zero_fraction"]}), flush=True)
            summary = model.run_arm(arrays, roles, tables, arm, arm_dir,
                {"diagnostic_contract_sha256": contract_sha, "role_manifest_sha256": contract["registration"]["sha256"],
                 "inputs": inputs}, progress=progress)
            verify_summary(summary, arm, contract_sha)
            audit = model.verify_artifacts(arm_dir, summary, arrays, roles, tables)
            if audit.get("passed") is not True:
                raise ValueError("Interaction saved-artifact replay failed")
            tracker.log_kernel("all", summary["aggregate"])
            exit_code = 0
        finally:
            tracker.finish(exit_code)
        tracking_path = output_dir / "tracking" / arm / "tracking.json"
        receipt = authenticated_json(tracking_path, sha256_file(tracking_path))
        if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("event_count") != 9 or receipt.get("tracking_errors") != 0
            or receipt.get("mode") != "offline" or receipt.get("stage") != "validation"
            or receipt.get("cloud_synced") is not False):
            raise ValueError("Offline interaction W&B recording incomplete")
        summaries[arm] = summary
        runs.append({"arm": arm, "summary_path": str((arm_dir / "summary.json").resolve()),
            "summary_sha256": sha256_file(arm_dir / "summary.json"), "tracking_path": str(tracking_path.resolve()),
            "tracking_sha256": sha256_file(tracking_path), "run_id": receipt["run_id"],
            "tracking_events": 9, "tracking_errors": 0, "cloud_synced": False, "artifact_audit": audit})
    if len({r["run_id"] for r in runs}) != 7:
        raise ValueError("Distinct interaction W&B run IDs required")
    validate_contract(contract_path, contract_sha)
    completed_at = datetime.now(timezone.utc).isoformat()
    suite = {"schema": SCHEMA, "completed": True, "completed_at": completed_at,
        "diagnostic_contract_path": str(contract_path.resolve()), "diagnostic_contract_sha256": contract_sha,
        "decision": screen(summaries), "runs": runs,
        "source_metrics": {arm: s["source_metrics"] for arm, s in summaries.items()},
        "pretraining_performed": False, "posttraining_performed": False, "promoted": False,
        "submission_performed": False, "new_raw_expression_read": False, "gpu_used": False,
        "download_performed": False, "cloud_synced": False, "tracking_events": 63, "tracking_errors": 0,
        "runtime": {"hostname": socket.gethostname(), "device": "cpu"}}
    dump_new(output_dir / "suite.json", suite)
    completion = {"schema": SCHEMA, "completed": True, "completed_at": completed_at,
        "suite_sha256": sha256_file(output_dir / "suite.json"), "diagnostic_contract_sha256": contract_sha,
        "screen_passed": suite["decision"]["passed"], "tracking_events": 63, "tracking_errors": 0,
        "cloud_synced": False, "promoted": False, "submission_performed": False}
    dump_new(output_dir / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--prior-v3-contract", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "register":
        if args.prior_v3_contract is None or args.protocol is None:
            parser.error("Registration requires --prior-v3-contract and --protocol")
        print(json.dumps({"contract_sha256": register(args.prior_v3_contract, args.protocol, args.output_dir)}))
        return 0
    if args.contract is None or not args.contract_sha256:
        parser.error("Execution requires --contract and --contract-sha256")
    return run(args.contract, args.contract_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
