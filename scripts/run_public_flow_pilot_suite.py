"""Run three fixed-budget, individually W&B-recorded GO ablation arms.

No raw data loading, automatic promotion, checkpoint selection or submission.
Use a scheduler allocation; all output directories must be fresh.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import socket
import re


def sha(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def arm_command(pilot, training_python, mode, *, device="cuda"):
    root = Path(__file__).resolve().parents[1]
    files = {"cache": pilot / "cache.npz", "contract": pilot / "contract.json",
             "source-manifest": pilot / "sources.json", "target-features": pilot / "go/features.npz",
             "feature-receipt": pilot / "go/receipt.json"}
    command = [str(training_python), str(root / "scripts/train_public_flow_pilot.py")]
    for name, path in files.items():
        if path.is_symlink() or not path.is_file():
            raise ValueError("Expected a regular pilot artifact")
        command.extend(["--" + name, str(path), "--" + name + "-sha256", sha(path)])
    command.extend(["--output-dir", str(pilot / "runs" / mode), "--feature-mode", mode,
                    "--device", device, "--stage", "pretrain", "--steps", "200", "--batch-size", "64",
                    "--hidden-size", "128", "--layers", "2", "--learning-rate", "0.0003",
                    "--weight-decay", "0.0001", "--seed", "20260909", "--integration-steps", "16",
                    "--shuffle-seed", "20260910"])
    return command


def verify_arm(pilot, mode):
    """A child exit zero is not proof of complete artifacts or W&B recording."""
    run = pilot / "runs" / mode
    tracking = pilot / "tracking" / mode
    for path in (run / "summary.json", run / "weights.npz", tracking / "tracking.json", tracking / "metrics.jsonl"):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Arm is missing a regular completion artifact")
    summary = json.loads((run / "summary.json").read_text())
    receipt = json.loads((tracking / "tracking.json").read_text())
    if (summary.get("status") != "completed_unpromoted_pilot" or summary.get("stage") != "pretrain"
            or summary.get("last_step") != 200 or summary.get("configuration", {}).get("feature_mode") != mode
            or summary.get("weights_sha256") != sha(run / "weights.npz")
            or summary.get("contract_sha256") != sha(pilot / "contract.json")
            or summary.get("source_manifest_sha256") != sha(pilot / "sources.json")
            or summary.get("provenance", {}).get("cache_sha256") != sha(pilot / "cache.npz")):
        raise ValueError("Arm completion summary does not match registered inputs")
    if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("tracking_errors") != 0 or receipt.get("mode") != "offline"
            or receipt.get("cloud_synced") is not False or receipt.get("event_count", 0) < 200):
        raise ValueError("W&B recording is incomplete")
    if not isinstance(receipt.get("run_id"), str) or re.fullmatch(r"[0-9a-f]{12}", receipt["run_id"]) is None:
        raise ValueError("Invalid W&B run identity")
    return {"summary_sha256": sha(run / "summary.json"), "tracking_receipt_sha256": sha(tracking / "tracking.json"),
            "run_id": receipt["run_id"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, required=True)
    parser.add_argument("--tracking-python", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURM_JOB_NODELIST"):
        raise RuntimeError("Submit through Slurm; never run on a login node")
    allocated = subprocess.check_output(["scontrol", "show", "hostnames", os.environ["SLURM_JOB_NODELIST"]], text=True).split()
    if socket.gethostname().split(".")[0] not in allocated:
        raise RuntimeError("This process must run on its allocated compute node")
    pilot = args.pilot_dir.resolve(strict=True)
    root = Path(__file__).resolve().parents[1]
    modes = ("true", "constant", "shuffled")
    # Resolve all inputs before creating any output or starting the first arm.
    commands = {mode: arm_command(pilot, Path(sys.executable), mode) for mode in modes}
    snapshot = {name: commands[modes[0]][commands[modes[0]].index("--" + name + "-sha256") + 1]
                for name in ("cache", "contract", "source-manifest", "target-features", "feature-receipt")}
    if any(os.path.lexists(pilot / name) for name in ("runs", "tracking", "suite.json")):
        raise FileExistsError("Suite outputs already exist; no automatic rerun")
    (pilot / "runs").mkdir(mode=0o700)
    (pilot / "tracking").mkdir(mode=0o700)
    outcomes, run_ids = [], set()
    for mode in modes:
        wrapped = [str(args.tracking_python), str(root / "scripts/run_training_with_wandb.py"),
                   "--stage", "pretrain", "--mode", "offline", "--project", "virtual-cell-challenge-2026",
                   "--group", "public-crispri-flow-h1-dev-v1", "--name", "go-" + mode,
                   "--output-dir", str(pilot / "tracking" / mode),
                   "--summary-file", str(pilot / "runs" / mode / "summary.json"), "--", *commands[mode]]
        code = subprocess.call(wrapped)
        evidence = {}
        if code == 0:
            try:
                evidence = verify_arm(pilot, mode)
                if evidence["run_id"] in run_ids:
                    raise ValueError("Reused W&B identity across independent arms")
                run_ids.add(evidence["run_id"])
            except (ValueError, OSError, TypeError):
                code = 78
                evidence = {"completion_evidence": "incomplete"}
        outcomes.append({"arm": mode, "exit_code": code, **evidence})
        print(json.dumps({"suite_arm": mode, "exit_code": code}), flush=True)
        if code != 0:
            break
    status = {"schema": "public-flow-pilot-suite-v1", "stage": "pretrain", "arms": outcomes,
              "completed": len(outcomes) == 3 and all(row["exit_code"] == 0 for row in outcomes),
              "slurm_job_id": os.environ["SLURM_JOB_ID"], "tracking_mode": "offline",
              "cloud_synced": False, "promotion_performed": False, "submission_performed": False,
              "input_sha256": snapshot, "contract_sha256": snapshot["contract"], "cache_sha256": snapshot["cache"]}
    with (pilot / "suite.json").open("x") as stream:
        json.dump(status, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return 0 if status["completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
