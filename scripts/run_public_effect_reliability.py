"""Freeze and execute an explicitly model-free public reliability diagnostic."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket

import public_effect_reliability as statistics
from public_reliability_tracking import METRICS, ReliabilityTracker
from prepare_public_control_anchor_v2 import load_cache
from train_public_flow_pilot import authenticated_bytes, authenticated_json, sha256_file, write_new

SCHEMA = "public-effect-reliability-run-v1"
FILES = {"cache": "cache.npz", "contract": "contract.json", "source_manifest": "sources.json",
         "prior_suite": "validation/suite.json"}
EXPECTED = {
    "cache_sha256": "13ed1ea60e0dff4d516098e9ef5c4d0520c7847ce2bdfc733082f645a5ee7186",
    "contract_sha256": "a94565b29d14d98588e8dbf58ad25481e1ebc5e20b209db643667d8a5344bbf1",
    "source_manifest_sha256": "b7e7d0a300fe4baeaf8d8963894ef2aa6b6681c1e8cb346f7e31c2515e7a78e7",
    "prior_suite_sha256": "c6eaae863a037bf0826d3f0384eeb781e4c4be29c1e094e66d5f9a25b917044f"}
MODULES = ("public_effect_reliability.py", "public_reliability_tracking.py", "wandb_training.py",
           "prepare_public_control_anchor_v2.py", "prepare_public_flow_pilot.py",
           "train_public_flow_pilot.py", "public_flow_conditioning.py", "public_crispri_flow.py",
           "build_go_target_features.py")
SOURCES = {"nadig_jurkat", "replogle_k562"}


def dump_new(path, value):
    write_new(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


def bindings(pilot):
    result = {name + "_sha256": sha256_file(pilot / relative) for name, relative in FILES.items()}
    if result != EXPECTED:
        raise ValueError("Only the exact exposed v2 inputs are registered for this diagnostic")
    return result


def register(pilot, protocol, output_dir):
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh registration directory required")
    inputs = bindings(pilot)  # Opaque hashes, no expression decoding or model fitting.
    record = {"schema": SCHEMA, "policy": statistics.policy(), "inputs": inputs,
        "created_at": datetime.now(timezone.utc).isoformat(), "stage": "validation",
        "registration_expression_decoded": False, "model_fitting_allowed": False,
        "new_raw_expression_allowed": False, "post_hoc_development_audit": True,
        "automatic_promotion": False, "input_paths": {name: str((pilot / relative).resolve()) for name, relative in FILES.items()},
        "protocol": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
        "code_sha256": {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "runner_sha256": sha256_file(Path(__file__))}
    output_dir.mkdir(mode=0o700)
    dump_new(output_dir / "contract.json", record)
    return sha256_file(output_dir / "contract.json")


def validate_registration(path, digest):
    contract = authenticated_json(path, digest)
    if (contract.get("schema") != SCHEMA or contract.get("policy") != statistics.policy()
            or contract.get("inputs") != EXPECTED or contract.get("stage") != "validation"
            or contract.get("registration_expression_decoded") is not False
            or contract.get("model_fitting_allowed") is not False
            or contract.get("new_raw_expression_allowed") is not False
            or contract.get("post_hoc_development_audit") is not True
            or contract.get("automatic_promotion") is not False):
        raise ValueError("Frozen diagnostic policy or known input identity differs")
    paths = contract.get("input_paths")
    if not isinstance(paths, dict) or set(paths) != set(FILES):
        raise ValueError("Exact input path roster required")
    for name, filename in paths.items():
        if sha256_file(Path(filename)) != EXPECTED[name + "_sha256"]:
            raise ValueError("Input bytes changed after registration")
    protocol = contract["protocol"]
    authenticated_bytes(Path(protocol["path"]), protocol["sha256"], 1 << 20)
    hashes = contract.get("code_sha256")
    if not isinstance(hashes, dict) or set(hashes) != set(MODULES):
        raise ValueError("Exact implementation roster required")
    for name, expected in hashes.items():
        if sha256_file(Path(__file__).with_name(name)) != expected:
            raise ValueError("Implementation changed after registration")
    if contract.get("runner_sha256") != sha256_file(Path(__file__)):
        raise ValueError("Runner changed after registration")
    return contract


def validate_aggregate(values):
    if not isinstance(values, dict) or not set(METRICS) <= set(values):
        raise ValueError("Missing core reliability statistics")
    for key in METRICS:
        value = values[key]
        if value is None and key.endswith("agreement"):
            continue
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("Nonfinite reliability scalar")
        if (key.endswith("energy") or key.endswith("disagreement_mse")) and value < 0:
            raise ValueError("Negative squared magnitude")
        if key.endswith("agreement") and not -1 <= value <= 1:
            raise ValueError("Unbounded normalized agreement")


def verify_report(report):
    validate_aggregate(report.get("aggregate"))
    sources = report.get("sources")
    if not isinstance(sources, list) or len(sources) != 2 or {s.get("source") for s in sources} != SOURCES:
        raise ValueError("Exactly two public sources required in report")
    for source in sources:
        validate_aggregate(source.get("aggregate"))
    seeds = report.get("seed_summaries")
    if not isinstance(seeds, list) or [item.get("seed") for item in seeds] != list(range(20260920, 20260952)):
        raise ValueError("Exactly the 32 frozen partitions must be reported")
    for item in seeds:
        validate_aggregate(item.get("aggregate"))
        source_rows = item.get("sources")
        if (not isinstance(source_rows, list) or len(source_rows) != 2
                or {s.get("source") for s in source_rows} != SOURCES):
            raise ValueError("Every partition requires both public sources")
        for source in source_rows:
            validate_aggregate(source.get("aggregate"))


def run(contract_path, contract_sha, output_dir):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Reliability computation requires an approved BioHPC compute host, never a head node")
    contract = validate_registration(contract_path, contract_sha)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh diagnostic output directory required")
    output_dir.mkdir(mode=0o700)
    hashes, paths = contract["inputs"], {k: Path(v) for k, v in contract["input_paths"].items()}
    tracker = ReliabilityTracker(output_dir / "tracking", config={**hashes,
        "diagnostic_contract_sha256": contract_sha, "entrypoint_sha256": contract["runner_sha256"]})
    exit_code = 1
    try:
        arrays, row_contract = load_cache(paths["cache"], hashes["cache_sha256"],
            paths["contract"], hashes["contract_sha256"], paths["source_manifest"], hashes["source_manifest_sha256"])
        if len(row_contract["excluded_targets"]) != 772:
            raise ValueError("Protected exclusions changed")
        report = statistics.compute_reliability(arrays)
        verify_report(report)
        for item in report["seed_summaries"]:
            tracker.log_reliability("all", item["aggregate"])
        for source in report["sources"]:
            tracker.log_reliability(source["source"], source["aggregate"])
        tracker.log_reliability("all", report["aggregate"])
        report.update(execution_schema=SCHEMA, execution_status="completed_model_free_diagnostic",
            diagnostic_contract_sha256=contract_sha, input_sha256=hashes,
            training_performed=False, posttraining_performed=False, promoted=False,
            submission_performed=False, new_raw_expression_read=False,
            cache_all_pools_integrity_validated=True,
            runtime={"hostname": socket.gethostname(), "device": "cpu"})
        dump_new(output_dir / "report.json", report)
        exit_code = 0
    finally:
        tracker.finish(exit_code)
    receipt = authenticated_json(output_dir / "tracking/tracking.json", sha256_file(output_dir / "tracking/tracking.json"))
    if (receipt.get("execution_status") != "completed" or receipt.get("training_exit_code") != 0
            or receipt.get("event_count") != 35 or receipt.get("tracking_errors") != 0
            or receipt.get("mode") != "offline" or receipt.get("stage") != "validation"
            or receipt.get("cloud_synced") is not False):
        raise ValueError("Offline W&B reliability recording incomplete")
    completion = {"schema": SCHEMA, "completed": True, "model_fitting_performed": False,
        "report_sha256": sha256_file(output_dir / "report.json"), "diagnostic_contract_sha256": contract_sha,
        "tracking_sha256": sha256_file(output_dir / "tracking/tracking.json"), "run_id": receipt["run_id"],
        "tracking_events": receipt["event_count"], "tracking_errors": 0, "cloud_synced": False,
        "automatic_promotion": False, "prior_v2_screen_changed": False}
    dump_new(output_dir / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run"))
    parser.add_argument("--pilot-dir", type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "register":
        if args.pilot_dir is None or args.protocol is None:
            parser.error("Registration needs --pilot-dir and --protocol")
        print(json.dumps({"diagnostic_contract_sha256": register(args.pilot_dir, args.protocol, args.output_dir)}))
        return 0
    if args.contract is None or not args.contract_sha256:
        parser.error("Execution requires the frozen contract and exact SHA")
    return run(args.contract, args.contract_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
