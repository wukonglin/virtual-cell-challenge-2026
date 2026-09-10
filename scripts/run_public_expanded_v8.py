"""Registered expanded public-cohort validation; no cloud, promotion or submission."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import resource
import socket
import stat
import time

import numpy as np

import prepare_public_expanded_v8 as preparation
import public_expanded_v8 as model
import public_replicated_cache_v7 as cache
import public_training_readiness as safe
import run_public_nested_v6 as prior
from public_expanded_tracking_v8 import ExpandedValidationTracker, GROUP
from public_nested_tracking_v6 import scalar_nested, SCOPES

SCHEMA = "public-expanded-regularization-execution-v8"
SUMMARY_SCHEMA = "public-expanded-regularization-summary-v8"
MODULES = tuple(dict.fromkeys(("run_public_expanded_v8.py", "public_expanded_v8.py",
    "prepare_public_expanded_v8.py", "public_expanded_tracking_v8.py", *preparation.MODULES,
    *prior.MODULES, "public_replicated_cache_v7.py", "prepare_public_replicated_v7.py")))
THREAD_VARIABLES = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
THREADS = 8
MAX_RSS = 16 << 30
MAX_DISK = 32 << 30
MAX_SECONDS = 4 * 3600
SOURCES = prior.SOURCES
METRICS = prior.METRICS
equal = prior.equal
dump_new = prior.dump_new
EVENTS = {arm: 9 if arm == "control" else 33 for arm in prior.model.ARMS}


def policy():
    return {"schema": SCHEMA, "model": model.policy(), "screen": prior.screen_policy(),
        "comparability": "same v6 implementation and screen; expanded evaluation cohort and group-ID-dependent donor ranking",
        "prior_public_development_exposure_acknowledged": True,
        "cpu_threads": THREADS, "maximum_peak_rss_bytes": MAX_RSS,
        "maximum_output_bytes": MAX_DISK, "maximum_elapsed_seconds": MAX_SECONDS,
        "runtime_limits": "thread preflight; RSS/disk/time checked between fits and after audits",
        "gpu_used": False, "stage": "validation", "wandb_mode": "offline", "wandb_group": GROUP,
        "events_by_arm": EVENTS, "total_events": 207,
        "fitted_two_head_models": 192, "control_references": 8,
        "raw_expression_read": False, "challenge_treated_used": False,
        "outer_response_selection": False, "automatic_promotion": False,
        "posttraining_performed": False, "submission_performed": False,
        "gate_adapter": "native v8 validation followed by explicit metadata projection to frozen v6 screen"}


def codes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in MODULES}


def register(role_path, role_sha, protocol, output_dir):
    role_path, protocol = Path(role_path).absolute(), Path(protocol).absolute()
    for path in (role_path, protocol):
        if path.is_symlink():
            raise ValueError("Nonsymlink registration/protocol required")
    record, roles, nested = preparation.validate_registration(role_path, role_sha)
    equal(record["protocol"], {"path": str(protocol.resolve()), "sha256": safe.digest(protocol)}, "protocol")
    require_roles(roles, nested)
    destination = Path(output_dir).absolute()
    if os.path.lexists(destination):
        raise FileExistsError("Fresh expanded execution directory required")
    value = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
        "registration": {"path": str(role_path.resolve()), "sha256": role_sha},
        "protocol": record["protocol"], "inputs": record["inputs"],
        "policy": policy(), "code_sha256": codes(), "expression_decoded": False}
    destination.mkdir(mode=0o700)
    dump_new(destination / "contract.json", value)
    return safe.digest(destination / "contract.json")


def require_roles(roles, nested):
    prior.expected_folds(roles)
    if len(roles["group_roster"]) != 1767 or len(nested["anchor_pools"]) != 101:
        raise ValueError("Exact expanded groups and reference pools required")
    if len(roles["targets"]) != 56 or len(roles["excluded_targets"]) != 772:
        raise ValueError("Expanded target/exclusion counts changed")


def validate_contract(path, sha):
    value = safe.read_json(Path(path), sha)
    equal(sorted(value), sorted(("schema", "created_at", "registration", "protocol", "inputs", "policy",
                       "code_sha256", "expression_decoded")), "execution fields")
    equal(value["schema"], SCHEMA, "execution schema")
    equal(value["policy"], policy(), "execution policy")
    equal(value["code_sha256"], codes(), "execution code")
    equal(value["expression_decoded"], False, "expression-free registration")
    created = datetime.fromisoformat(value["created_at"])
    if created.tzinfo is None:
        raise ValueError("Timezone-aware registration time required")
    binding = value["registration"]
    equal(sorted(binding), ["path", "sha256"], "registration binding fields")
    record, roles, nested = preparation.validate_registration(Path(binding["path"]), binding["sha256"])
    equal(value["inputs"], record["inputs"], "expanded input bindings")
    equal(value["protocol"], record["protocol"], "protocol binding")
    safe.binding(value["protocol"], json_document=False)
    require_roles(roles, nested)
    return value, record, roles, nested


def provenance(contract, sha):
    return {"diagnostic_contract_sha256": sha,
            "role_manifest_sha256": contract["registration"]["sha256"], "inputs": contract["inputs"]}


def screen_projection(summary):
    """Only adapt version metadata; never mutate reports, metric values or thresholds."""
    equal(summary.get("schema"), SUMMARY_SCHEMA, "expanded summary schema")
    equal(summary.get("policy"), model.policy(), "expanded model policy")
    return {**summary, "schema": "public-nested-regularization-summary-v6", "policy": prior.model.policy()}


def verify_summary(summary, arm, contract_sha=None, *, contract=None, roles=None, nested=None):
    projected = screen_projection(summary)
    if contract is not None:
        equal(summary.get("provenance"), provenance(contract, contract_sha), "expanded summary provenance")
    prior.verify_summary(projected, arm, contract_sha, roles=roles, nested=nested)


def screen(summaries):
    return prior.screen({arm: screen_projection(value) for arm, value in summaries.items()})


def verify_audit(audit, arm):
    for key in (*prior.AUDIT_TRUE, "sharded_artifacts_verified", "exact_artifact_roster_verified",
                "shard_headers_verified_before_decode", "same_open_checksum_verified", "output_budget_verified"):
        equal(audit.get(key), True, "artifact audit " + key)
    for key, expected in {"ridge_refits": 0, "inner_candidates_replayed": 0 if arm == "control" else 24,
            "out_of_fold_groups": 1767, "out_of_fold_source_targets": 112,
            "fits_replayed": 8 if arm == "control" else 32,
            "npz_shards_verified": 16 if arm == "control" else 64}.items():
        equal(audit.get(key), expected, "artifact audit " + key)
    for key in ("shard_compressed_bytes", "shard_expanded_bytes"):
        value = audit.get(key)
        if type(value) is not int or value <= 0 or value > MAX_DISK:
            raise ValueError("Invalid artifact audit size")


def runtime_preflight():
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Use an approved BioHPC compute host, never a login/head node")
    for name in THREAD_VARIABLES:
        if os.environ.get(name) != str(THREADS):
            raise RuntimeError("Set all numerical thread variables to the registered eight threads")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("This CPU validation requires CUDA_VISIBLE_DEVICES empty")
    from threadpoolctl import threadpool_info
    if any(pool.get("num_threads", 0) > THREADS for pool in threadpool_info()):
        raise RuntimeError("Active numerical library exceeds the registered CPU thread bound")


def resources(output_dir, started):
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    size = 0
    for root, directories, files in os.walk(output_dir, followlinks=False):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("Unexpected symlink output directory")
        for name in files:
            path = Path(root) / name
            if path.is_symlink():
                raise ValueError("Unexpected symlink output file")
            info = path.stat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("Unexpected nonregular output file")
            size += info.st_size
    elapsed = time.monotonic() - started
    if peak > MAX_RSS or size > MAX_DISK or elapsed > MAX_SECONDS:
        raise RuntimeError("Expanded validation exceeded its registered RSS/disk/time budget")
    return {"peak_rss_bytes": peak, "output_bytes": size, "elapsed_seconds": elapsed,
            "cpu_threads": THREADS, "device": "cpu", "hostname": socket.gethostname()}


def expected_journal(summary, arm):
    """Reconstruct actual scalar events from the saved, audited fit reports."""
    rows = []
    def append(phase, scope, metrics, training=None, regularization=None):
        values = scalar_nested(phase, scope, metrics, training)
        if not values:
            raise ValueError("Missing scalar event metrics")
        if regularization is not None:
            values[f"validation/{phase}/{SCOPES[scope]}/regularization"] = regularization
        index = len(rows) + 1
        rows.append({"event_index": index, "analysis/event_index": index, **values})
    for report in summary["folds"]:
        source = report["held_source"]
        if arm != "control":
            for candidate in report["inner_selection"]["candidates"]:
                append("inner_regularization", source, candidate["metrics"],
                       candidate["training_diagnostics"], candidate["regularization"])
        append("nested_outer", source, report["metrics"], report["training_diagnostics"],
               None if arm == "control" else report["regularization"])
    append("nested_outer", "all", summary["aggregate"])
    return rows


def verify_tracking(path, arm, contract_sha, summary):
    result = safe.read_json(path, safe.digest(path))
    for key, expected in {"execution_status": "completed", "training_exit_code": 0,
            "event_count": EVENTS[arm], "tracking_errors": 0, "mode": "offline",
            "stage": "validation", "group": GROUP, "cloud_synced": False,
            "raw_artifacts_uploaded": False, "scientific_quality_inferred_from_exit": False}.items():
        equal(result.get(key), expected, "tracking " + key)
    equal(result.get("config", {}).get("diagnostic_contract_sha256"), contract_sha, "tracking contract")
    journal = path.parent / "metrics.jsonl"
    with safe.regular_reader(journal) as handle:
        payload = handle.read(8 << 20)
        if handle.read(1):
            raise ValueError("Oversized tracking journal")
    rows = [json.loads(line) for line in payload.splitlines()]
    equal([row["event_index"] for row in rows], list(range(1, EVENTS[arm] + 1)), "tracking event roster")
    equal(rows, expected_journal(summary, arm), "tracking scalar event replay")
    return result


def run(contract_path, contract_sha, output_dir):
    runtime_preflight()
    started = time.monotonic()
    contract, record, roles, nested = validate_contract(contract_path, contract_sha)
    output_dir = Path(output_dir).absolute()
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh expanded suite output required")
    output_dir.mkdir(mode=0o700)
    outer_tables, inner_tables = preparation.load_feature_tables(record)
    paths, inputs = record["input_paths"], record["inputs"]
    arrays, row_contract = cache.load_cache(Path(paths["cache"]), inputs["cache_sha256"],
        Path(paths["contract"]), inputs["contract_sha256"], Path(paths["source_manifest"]), inputs["source_manifest_sha256"])
    equal(row_contract["excluded_targets"], roles["excluded_targets"], "protected exclusions")
    for value in arrays.values():
        value.setflags(write=False)
    summaries, runs = {}, []
    (output_dir / "runs").mkdir(mode=0o700)
    for arm in prior.model.ARMS:
        validate_contract(contract_path, contract_sha)
        resources(output_dir, started)
        tracker = ExpandedValidationTracker(output_dir / "tracking" / arm, arm=arm,
            config={**inputs, "diagnostic_contract_sha256": contract_sha,
                    "split_sha256": contract["registration"]["sha256"],
                    "entrypoint_sha256": contract["code_sha256"]["run_public_expanded_v8.py"]})
        exit_code = 1
        arm_dir = output_dir / "runs" / arm
        try:
            def progress(event):
                resources(output_dir, started)
                if event["phase"] == "inner":
                    tracker.log_inner(event["held_outer_source"], event["regularization"],
                        event["metrics"], event["training_diagnostics"])
                else:
                    tracker.log_outer(event["held_outer_source"], event["metrics"], event["training_diagnostics"],
                        regularization=None if arm == "control" else event["regularization"])
                print(json.dumps({"arm": arm, "phase": event["phase"], "fold": event["fold_index"] + 1,
                    "held_outer_source": event["held_outer_source"], "lambda": event["regularization"],
                    "primary_mse": event["metrics"]["target_pooled_mse"]}), flush=True)
            summary = model.run_arm(arrays, roles, outer_tables, nested, inner_tables, arm, arm_dir,
                                    provenance(contract, contract_sha), progress=progress)
            verify_summary(summary, arm, contract_sha, contract=contract, roles=roles, nested=nested)
            audit = model.audit_arm(arm_dir, summary, arrays, roles, outer_tables, nested, inner_tables,
                                    provenance(contract, contract_sha))
            verify_audit(audit, arm)
            tracker.log_outer("all", summary["aggregate"])
            exit_code = 0
        finally:
            tracker.finish(exit_code)
        tp = output_dir / "tracking" / arm / "tracking.json"
        receipt = verify_tracking(tp, arm, contract_sha, summary)
        summaries[arm] = summary
        runs.append({"arm": arm, "summary_path": str((arm_dir / "summary.json").resolve()),
            "summary_sha256": safe.digest(arm_dir / "summary.json"), "tracking_path": str(tp.resolve()),
            "tracking_sha256": safe.digest(tp), "journal_sha256": safe.digest(tp.parent / "metrics.jsonl"),
            "run_id": receipt["run_id"], "tracking_events": EVENTS[arm], "tracking_errors": 0,
            "cloud_synced": False, "artifact_audit": audit})
        print(json.dumps({"arm": arm, "status": "saved_artifacts_replayed", **resources(output_dir, started)}), flush=True)
    if len({row["run_id"] for row in runs}) != 7:
        raise ValueError("Distinct W&B run IDs required")
    validate_contract(contract_path, contract_sha)
    completed = datetime.now(timezone.utc).isoformat()
    decision = screen(summaries)
    suite = {"schema": SCHEMA, "completed": True, "completed_at": completed,
        "diagnostic_contract_path": str(Path(contract_path).resolve()), "diagnostic_contract_sha256": contract_sha,
        "decision": decision, "runs": runs, "source_metrics": {a: s["source_metrics"] for a, s in summaries.items()},
        "model_fitting_performed": True, "fitted_two_head_models": 192, "control_references": 8,
        "pretraining_performed": False, "posttraining_performed": False, "promoted": False,
        "submission_performed": False, "new_raw_expression_read": False, "challenge_treated_used": False,
        "gpu_used": False, "download_performed": False, "cloud_synced": False,
        "tracking_events": 207, "tracking_errors": 0, "runtime": resources(output_dir, started),
        "next_stage": {"public_screen_passed": decision["passed"], "automatic_launch": False,
            "promoter_ablation_needs_new_registration": True, "feng_ready": False,
            "compatible_flow_parent_available": False, "submission_ready": False,
            "full_gene_raw_count_emitter_available": False}}
    dump_new(output_dir / "suite.json", suite)
    completion = {"schema": SCHEMA, "completed": True, "completed_at": completed,
        "suite_sha256": safe.digest(output_dir / "suite.json"), "diagnostic_contract_sha256": contract_sha,
        "screen_passed": decision["passed"], "tracking_events": 207, "tracking_errors": 0,
        "cloud_synced": False, "promoted": False, "submission_performed": False}
    resources(output_dir, started)
    dump_new(output_dir / "complete.json", completion)
    print(json.dumps(completion, sort_keys=True), flush=True)
    return 0


def verify_suite(suite_path, suite_sha):
    """Replay a completed suite from frozen inputs without fitting or W&B init."""
    runtime_preflight()
    suite_path = Path(suite_path).absolute()
    if suite_path.name != "suite.json" or suite_path.is_symlink():
        raise ValueError("Canonical nonsymlink suite.json required")
    suite = safe.read_json(suite_path, suite_sha)
    equal(sorted(suite), sorted(("schema", "completed", "completed_at", "diagnostic_contract_path",
        "diagnostic_contract_sha256", "decision", "runs", "source_metrics", "model_fitting_performed",
        "fitted_two_head_models", "control_references", "pretraining_performed", "posttraining_performed",
        "promoted", "submission_performed", "new_raw_expression_read", "challenge_treated_used", "gpu_used",
        "download_performed", "cloud_synced", "tracking_events", "tracking_errors", "runtime", "next_stage")),
        "suite fields")
    for key, expected in {"schema": SCHEMA, "completed": True, "model_fitting_performed": True,
            "fitted_two_head_models": 192, "control_references": 8, "pretraining_performed": False,
            "posttraining_performed": False, "promoted": False, "submission_performed": False,
            "new_raw_expression_read": False, "challenge_treated_used": False, "gpu_used": False,
            "download_performed": False, "cloud_synced": False, "tracking_events": 207, "tracking_errors": 0}.items():
        equal(suite.get(key), expected, "suite " + key)
    contract_sha = suite["diagnostic_contract_sha256"]
    contract, record, roles, nested = validate_contract(Path(suite["diagnostic_contract_path"]), contract_sha)
    completed = datetime.fromisoformat(suite["completed_at"])
    created = datetime.fromisoformat(contract["created_at"])
    if (completed.tzinfo is None or completed < created
            or (completed - datetime.now(timezone.utc)).total_seconds() > 300):
        raise ValueError("Invalid suite completion timestamp")
    runtime = suite["runtime"]
    equal(sorted(runtime), sorted(("peak_rss_bytes", "output_bytes", "elapsed_seconds", "cpu_threads", "device", "hostname")),
          "saved runtime fields")
    equal(runtime["cpu_threads"], THREADS, "saved CPU threads")
    equal(runtime["device"], "cpu", "saved device")
    if (not isinstance(runtime["hostname"], str)
            or runtime["hostname"].split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}):
        raise ValueError("Invalid saved compute host")
    for key, maximum in (("peak_rss_bytes", MAX_RSS), ("output_bytes", MAX_DISK)):
        if type(runtime[key]) is not int or not 0 < runtime[key] <= maximum:
            raise ValueError("Invalid saved runtime byte budget")
    elapsed = runtime["elapsed_seconds"]
    if type(elapsed) not in (float, int) or not math.isfinite(elapsed) or not 0 <= elapsed <= MAX_SECONDS:
        raise ValueError("Invalid saved elapsed budget")
    complete = safe.read_json(suite_path.with_name("complete.json"))
    expected_complete = {"schema": SCHEMA, "completed": True, "completed_at": suite["completed_at"],
        "suite_sha256": suite_sha, "diagnostic_contract_sha256": contract_sha,
        "screen_passed": suite["decision"]["passed"], "tracking_events": 207, "tracking_errors": 0,
        "cloud_synced": False, "promoted": False, "submission_performed": False}
    equal(complete, expected_complete, "suite completion")
    equal([r["arm"] for r in suite["runs"]], list(prior.model.ARMS), "suite arm roster")
    if len({r["run_id"] for r in suite["runs"]}) != 7:
        raise ValueError("Distinct saved tracking runs required")
    outer_tables, inner_tables = preparation.load_feature_tables(record)
    paths, inputs = record["input_paths"], record["inputs"]
    arrays, _ = cache.load_cache(Path(paths["cache"]), inputs["cache_sha256"],
        Path(paths["contract"]), inputs["contract_sha256"], Path(paths["source_manifest"]), inputs["source_manifest_sha256"])
    summaries = {}
    for item in suite["runs"]:
        equal(sorted(item), sorted(("arm", "summary_path", "summary_sha256", "tracking_path", "tracking_sha256",
            "journal_sha256", "run_id", "tracking_events", "tracking_errors", "cloud_synced", "artifact_audit")),
            "suite run fields")
        arm = item["arm"]
        arm_dir = suite_path.parent / "runs" / arm
        tp = suite_path.parent / "tracking" / arm / "tracking.json"
        equal(item["summary_path"], str((arm_dir / "summary.json").resolve()), "suite summary path")
        equal(item["tracking_path"], str(tp.resolve()), "suite tracking path")
        summary = safe.read_json(arm_dir / "summary.json", item["summary_sha256"])
        verify_summary(summary, arm, contract_sha, contract=contract, roles=roles, nested=nested)
        equal(safe.digest(tp), item["tracking_sha256"], "tracking receipt SHA")
        equal(safe.digest(tp.parent / "metrics.jsonl"), item["journal_sha256"], "tracking journal SHA")
        tracking = verify_tracking(tp, arm, contract_sha, summary)
        equal(item["run_id"], tracking["run_id"], "tracking run ID")
        for key, expected in {"tracking_events": EVENTS[arm], "tracking_errors": 0, "cloud_synced": False}.items():
            equal(item.get(key), expected, "suite run " + key)
        audit = model.audit_arm(arm_dir, summary, arrays, roles, outer_tables, nested, inner_tables,
                                provenance(contract, contract_sha))
        verify_audit(audit, arm)
        equal(item["artifact_audit"], audit, "saved artifact audit replay")
        summaries[arm] = summary
        print(json.dumps({"arm": arm, "verified_without_refitting": True}), flush=True)
    decision = screen(summaries)
    equal(suite["decision"], decision, "recomputed scientific screen")
    equal(suite["source_metrics"], {a: s["source_metrics"] for a, s in summaries.items()}, "suite source metrics")
    equal(suite["next_stage"], {"public_screen_passed": decision["passed"], "automatic_launch": False,
        "promoter_ablation_needs_new_registration": True, "feng_ready": False,
        "compatible_flow_parent_available": False, "submission_ready": False,
        "full_gene_raw_count_emitter_available": False}, "progression gate")
    validate_contract(Path(suite["diagnostic_contract_path"]), contract_sha)
    result = {"schema": "public-expanded-verification-v8", "verified": True,
        "suite_sha256": suite_sha, "groups": 1767, "saved_fits_replayed": 192,
        "control_references_replayed": 8, "ridge_refits": 0, "tracking_events_replayed": 207,
        "conditioning_passed": decision["conditioning_passed"],
        "distribution_passed": decision["distribution_passed"], "safety_passed": decision["safety_passed"],
        "feng_ready": False, "submission_ready": False}
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("register", "run", "verify"))
    parser.add_argument("--role-manifest", type=Path)
    parser.add_argument("--role-manifest-sha256")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--contract", type=Path)
    parser.add_argument("--contract-sha256")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--suite", type=Path)
    parser.add_argument("--suite-sha256")
    args = parser.parse_args(argv)
    if args.action == "verify":
        if args.suite is None or not args.suite_sha256:
            parser.error("Verification requires suite and SHA")
        verify_suite(args.suite, args.suite_sha256)
        return 0
    if args.output_dir is None:
        parser.error("Registration/execution requires fresh output directory")
    if args.action == "register":
        if args.role_manifest is None or not args.role_manifest_sha256 or args.protocol is None:
            parser.error("Registration requires roles, SHA and protocol")
        print(json.dumps({"contract_sha256": register(args.role_manifest, args.role_manifest_sha256,
                                                       args.protocol, args.output_dir)}))
        return 0
    if args.contract is None or not args.contract_sha256:
        parser.error("Execution requires contract and SHA")
    return run(args.contract, args.contract_sha256, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
