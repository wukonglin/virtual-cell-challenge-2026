"""Authenticate and run a fixed four-arm, two-source public validation suite.

Registration hashes opaque cache bytes, never decodes expression. Numeric work
runs only on an approved BioHPC compute host. All outputs are fresh and unpromoted.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

import public_control_anchor_v2 as baseline
from train_public_flow_pilot import authenticated_json, sha256_file, write_new

SCHEMA = "public-control-anchor-suite-v2"
FILES = {"cache": "cache.npz", "target-features": "go/features.npz",
         "feature-receipt": "go/receipt.json", "contract": "contract.json",
         "source-manifest": "sources.json"}
MODULES = ("public_control_anchor_v2.py", "prepare_public_control_anchor_v2.py",
           "public_flow_stable_conditioning.py", "public_flow_conditioning.py",
           "public_mean_effect_baseline.py", "train_public_flow_pilot.py",
           "build_go_target_features.py", "public_crispri_flow.py", "prepare_public_flow_pilot.py")
TRACKING = ("wandb_training.py", "run_training_with_wandb.py")
SOURCES = {"replogle_k562", "nadig_jurkat"}


def dump_new(path, payload):
    write_new(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def bindings(pilot):
    result = {}
    for name, relative in FILES.items():
        path = pilot / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Regular versioned pilot inputs required")
        result[name.replace("-", "_") + "_sha256"] = sha256_file(path)
    return result


def screening_policy():
    return {"minimum_relative_mse_improvement_vs_control": 0.01,
            "minimum_relative_mse_improvement_vs_shuffled": 0.01,
            "strictly_better_mse_than_context": True,
            "noninferior_mmd_vs_control_context_shuffled": True,
            "required_sources": sorted(SOURCES), "both_sources_must_pass": True,
            "interpretation": "descriptive engineering effect-size screen; not significance or unseen-target evidence",
            "automatic_promotion": False}


def register(pilot, output):
    contract = {"schema": SCHEMA, "policy": baseline.policy(), "inputs": bindings(pilot),
        "created_at": datetime.now(timezone.utc).isoformat(), "screen": screening_policy(),
        "registration_npz_decoded": False, "stage": "validation",
        "prior_project_source_exposure_acknowledged": True,
        "challenge_2026_treated_allowed": False, "h1_rpe1_feng_expression_allowed": False,
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "tracking_code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in TRACKING},
        "suite_code_sha256": sha256_file(Path(__file__))}
    dump_new(output, contract)
    return sha256_file(output)


def validate_diagnostic_contract(path, expected_sha, inputs):
    contract = authenticated_json(path, expected_sha)
    if (contract.get("schema") != SCHEMA or contract.get("policy") != baseline.policy()
            or contract.get("inputs") != inputs or contract.get("screen") != screening_policy()
            or contract.get("stage") != "validation" or contract.get("registration_npz_decoded") is not False
            or contract.get("challenge_2026_treated_allowed") is not False
            or contract.get("h1_rpe1_feng_expression_allowed") is not False):
        raise ValueError("Diagnostic contract policy or input bindings mismatch")
    for field, names in (("code_sha256", MODULES), ("tracking_code_sha256", TRACKING)):
        hashes = contract.get(field)
        if not isinstance(hashes, dict) or set(hashes) != set(names):
            raise ValueError("Exact runtime code roster required")
        for name, digest in hashes.items():
            if sha256_file(Path(__file__).with_name(name)) != digest:
                raise ValueError("Runtime code changed after registration")
    if contract.get("suite_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Suite code changed after registration")
    return contract


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def checked_metrics(values):
    required = ("model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
                "negative_fraction", "projection_fraction")
    if not isinstance(values, dict) or any(not finite_number(values.get(k)) for k in required):
        raise ValueError("Missing finite aggregate validation diagnostics")
    if any(values[k] < 0 for k in ("model_centroid_mse", "control_centroid_mse")):
        raise ValueError("Negative squared error")
    if any(not 0 <= values[k] <= 1 for k in ("negative_fraction", "projection_fraction")):
        raise ValueError("Invalid aggregate fraction")
    return values


def verify_arm(output, arm, contract_sha):
    run, tracking = output / "runs" / arm, output / "tracking" / arm
    summary = authenticated_json(run / "summary.json", sha256_file(run / "summary.json"))
    receipt = authenticated_json(tracking / "tracking.json", sha256_file(tracking / "tracking.json"))
    if (summary.get("schema") != "public-control-anchor-validation-summary-v2"
            or summary.get("status") != "completed_unpromoted_diagnostic" or summary.get("arm") != arm
            or summary.get("diagnostic_contract_sha256") != contract_sha
            or summary.get("policy") != baseline.policy() or summary.get("stage") != "validation"
            or summary.get("posttraining_performed") is not False
            or summary.get("development_used_for_fit_or_selection") is not False
            or summary.get("count_emitter") is not False
            or summary.get("exact_ntc_identity_verified") is not True
            or summary.get("exact_zero_effect_identity_verified") is not True):
        raise ValueError("Incomplete or mismatched control-anchor summary")
    for name in ("weights", "predictions"):
        if summary.get(name + "_sha256") != sha256_file(run / (name + ".npz")):
            raise ValueError("Output hash mismatch")
    folds = summary.get("outer_folds")
    if (not isinstance(folds, list) or len(folds) != 2
            or {item.get("held_source") for item in folds} != SOURCES):
        raise ValueError("Two exact public source holdouts required")
    for fold in folds:
        if set(fold.get("train_sources", [])) != SOURCES - {fold["held_source"]}:
            raise ValueError("Source leakage in reported fold")
        checked_metrics(fold.get("aggregate"))
    metrics = checked_metrics(summary["deployment_rollout_diagnostics"]["kind_metrics"]["context"])
    if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("tracking_errors") != 0 or receipt.get("event_count", 0) < 3
            or receipt.get("mode") != "offline" or receipt.get("stage") != "validation"
            or receipt.get("cloud_synced") is not False):
        raise ValueError("Incomplete offline W&B validation recording")
    return {"arm": arm, "summary_sha256": sha256_file(run / "summary.json"),
            "tracking_sha256": sha256_file(tracking / "tracking.json"), "run_id": receipt["run_id"],
            "metrics": metrics, "outer_folds": folds,
            "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True}


def screen(arms):
    if len(arms) != len(baseline.ARMS) or {a["arm"] for a in arms} != set(baseline.ARMS):
        raise ValueError("Every fixed arm must complete before screening")
    lookup = {a["arm"]: {f["held_source"]: checked_metrics(f["aggregate"])
                         for f in a["outer_folds"]} for a in arms}
    if any(set(folds) != SOURCES for folds in lookup.values()):
        raise ValueError("Screen requires identical held-source roster")
    reports = []
    for source in sorted(SOURCES):
        values = {arm: folds[source] for arm, folds in lookup.items()}
        true = values["true"]
        control_mse = values["control"]["model_centroid_mse"]
        if any(not math.isclose(v["control_centroid_mse"], control_mse, rel_tol=1e-12, abs_tol=1e-12)
               for v in values.values()):
            raise ValueError("Arms did not use the same control reference")
        checks = {}
        for comparator in ("control", "shuffled"):
            reference = values[comparator]["model_centroid_mse"]
            checks["mse_gain_vs_" + comparator] = bool(reference > 0 and true["model_centroid_mse"] <= 0.99 * reference)
        checks["mse_better_than_context"] = true["model_centroid_mse"] < values["context"]["model_centroid_mse"]
        checks["mmd_noninferior"] = all(true["model_mmd2"] <= values[a]["model_mmd2"]
                                               for a in ("control", "context", "shuffled"))
        checks["all_arms_nonnegative"] = all(v["negative_fraction"] == 0 for v in values.values())
        reports.append({"held_source": source, "checks": checks, "passed": all(checks.values())})
    return {"policy": screening_policy(), "source_results": reports,
            "passed": all(item["passed"] for item in reports),
            "promoted": False, "submission_performed": False, "posttraining_performed": False}


def run(pilot, contract_path, contract_sha, output):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("This bounded CPU suite requires an approved BioHPC compute host")
    contract = validate_diagnostic_contract(contract_path, contract_sha, bindings(pilot))
    if os.path.lexists(output):
        raise FileExistsError("Suite outputs must be fresh")
    output.mkdir(mode=0o700)
    (output / "runs").mkdir()
    (output / "tracking").mkdir()
    outcomes, run_ids = [], set()
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")
    for arm in baseline.ARMS:
        command = [sys.executable, str(Path(__file__).with_name("public_control_anchor_v2.py"))]
        for name, relative in FILES.items():
            command += ["--" + name, str(pilot / relative), "--" + name + "-sha256",
                        contract["inputs"][name.replace("-", "_") + "_sha256"]]
        command += ["--diagnostic-contract", str(contract_path), "--diagnostic-contract-sha256", contract_sha,
                    "--arm", arm, "--output-dir", str(output / "runs" / arm)]
        wrapped = [sys.executable, str(Path(__file__).with_name("run_training_with_wandb.py")),
                   "--stage", "validation", "--mode", "offline", "--group", "public-control-anchor-v2",
                   "--name", "loso-" + arm, "--output-dir", str(output / "tracking" / arm),
                   "--summary-file", str(output / "runs" / arm / "summary.json"), "--", *command]
        code = subprocess.call(wrapped, env=env)
        if code:
            dump_new(output / "suite.json", {"schema": SCHEMA, "completed": False, "failed_arm": arm,
                "child_exit_code": code, "arms": outcomes, "diagnostic_contract_sha256": contract_sha})
            return code
        evidence = verify_arm(output, arm, contract_sha)
        if evidence["run_id"] in run_ids:
            raise ValueError("Reused W&B identity")
        run_ids.add(evidence["run_id"])
        outcomes.append(evidence)
    dump_new(output / "suite.json", {"schema": SCHEMA, "completed": True,
        "arms": outcomes, "diagnostic_contract_sha256": contract_sha,
        "screen": screen(outcomes), "cloud_synced": False, "promoted": False,
        "posttraining_performed": False, "submission_performed": False})
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-contract", type=Path, required=True)
    parser.add_argument("--diagnostic-contract-sha256")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.action == "register":
        print(json.dumps({"diagnostic_contract_sha256": register(args.pilot_dir, args.diagnostic_contract)}))
        return 0
    if not args.diagnostic_contract_sha256 or args.output_dir is None:
        parser.error("Run requires explicit contract SHA and fresh output directory")
    return run(args.pilot_dir, args.diagnostic_contract, args.diagnostic_contract_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
