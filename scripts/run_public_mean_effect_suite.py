"""Freeze then run a small public-only diagnostic suite with offline W&B.

No GPU allocation, raw expression access, model promotion or external submission.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import public_mean_effect_baseline as baseline
from train_public_flow_pilot import authenticated_json, sha256_file, write_new

FILES = {"cache": "cache.npz", "target-features": "go/features.npz",
         "feature-receipt": "go/receipt.json", "contract": "contract.json",
         "source-manifest": "sources.json"}
MODULES = ("public_mean_effect_baseline.py", "train_public_flow_pilot.py",
           "public_flow_conditioning.py", "build_go_target_features.py", "public_crispri_flow.py")


def dump_new(path, payload):
    write_new(path, (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def bindings(pilot):
    result = {}
    for name, relative in FILES.items():
        path = pilot / relative
        if path.is_symlink() or not path.is_file():
            raise ValueError("Regular pilot inputs required")
        result[name.replace("-", "_") + "_sha256"] = sha256_file(path)
    return result


def register(pilot, output):
    # Hash opaque cache/feature files; do not decode expression here.
    contract = {"schema": baseline.SCHEMA, "policy": baseline.policy(), "inputs": bindings(pilot),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prior_development_exposure_acknowledged": True, "challenge_2026_treated_allowed": False,
        "registration_npz_decoded": False,
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "suite_code_sha256": sha256_file(Path(__file__)),
        "tracking_code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in
                                  ("wandb_training.py", "run_training_with_wandb.py")}}
    dump_new(output, contract)
    return sha256_file(output)


def verify_arm(output, arm, contract_sha):
    run, tracking = output / "runs" / arm, output / "tracking" / arm
    summary = authenticated_json(run / "summary.json", sha256_file(run / "summary.json"))
    receipt = authenticated_json(tracking / "tracking.json", sha256_file(tracking / "tracking.json"))
    if (summary.get("status") != "completed_unpromoted_diagnostic" or summary.get("arm") != arm
            or summary.get("provenance", {}).get("diagnostic_contract_sha256") != contract_sha
            or summary.get("policy") != baseline.policy()
            or summary.get("posttraining_performed") is not False
            or summary.get("development_used_for_fit_or_selection") is not False):
        raise ValueError("Incomplete or mismatched baseline summary")
    for name in ("weights", "predictions"):
        if summary.get(name + "_sha256") != sha256_file(run / (name + ".npz")):
            raise ValueError("Output hash mismatch")
    if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("tracking_errors") != 0 or receipt.get("event_count", 0) < 3
            or receipt.get("mode") != "offline" or receipt.get("cloud_synced") is not False):
        raise ValueError("Incomplete offline W&B recording")
    return {"arm": arm, "summary_sha256": sha256_file(run / "summary.json"),
            "tracking_sha256": sha256_file(tracking / "tracking.json"), "run_id": receipt["run_id"],
            "metrics": summary["deployment_rollout_diagnostics"]["kind_metrics"]["context"]}


def run(pilot, contract_path, contract_sha, output):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("This bounded CPU suite requires an approved BioHPC compute host")
    contract = baseline.validate_diagnostic_contract(contract_path, contract_sha, bindings(pilot))
    if contract.get("suite_code_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Suite code changed after registration")
    tracking_codes = contract.get("tracking_code_sha256")
    if not isinstance(tracking_codes, dict) or set(tracking_codes) != {"wandb_training.py", "run_training_with_wandb.py"}:
        raise ValueError("Exact tracking code roster required")
    for name, expected in tracking_codes.items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError("Tracking code changed after registration")
    if os.path.lexists(output):
        raise FileExistsError("Suite outputs must be fresh")
    output.mkdir(mode=0o700)
    (output / "runs").mkdir()
    (output / "tracking").mkdir()
    outcomes, run_ids = [], set()
    for arm in baseline.ARMS:
        command = [sys.executable, str(Path(__file__).with_name("public_mean_effect_baseline.py"))]
        for name, relative in FILES.items():
            command += ["--" + name, str(pilot / relative), "--" + name + "-sha256",
                        contract["inputs"][name.replace("-", "_") + "_sha256"]]
        command += ["--diagnostic-contract", str(contract_path), "--diagnostic-contract-sha256", contract_sha,
                    "--arm", arm, "--output-dir", str(output / "runs" / arm)]
        wrapped = [sys.executable, str(Path(__file__).with_name("run_training_with_wandb.py")),
                   "--stage", "pretrain", "--mode", "offline", "--group", "public-mean-effect-diagnostic-v1",
                   "--name", "residual-" + arm, "--output-dir", str(output / "tracking" / arm),
                   "--summary-file", str(output / "runs" / arm / "summary.json"), "--", *command]
        code = subprocess.call(wrapped)
        if code:
            dump_new(output / "suite.json", {"completed": False, "failed_arm": arm,
                     "child_exit_code": code, "arms": outcomes, "diagnostic_contract_sha256": contract_sha})
            return code
        evidence = verify_arm(output, arm, contract_sha)
        if evidence["run_id"] in run_ids:
            raise ValueError("Reused W&B identity")
        run_ids.add(evidence["run_id"])
        outcomes.append(evidence)
    dump_new(output / "suite.json", {"schema": "public-mean-effect-suite-v1", "completed": True,
        "arms": outcomes, "diagnostic_contract_sha256": contract_sha, "cloud_synced": False,
        "promoted": False, "posttraining_performed": False, "submission_performed": False})
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
