"""Expanded runner contracts, frozen scientific gates, bounds and orchestration."""
import copy
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_expanded_v8 as runner
from public_expanded_tracking_v8 import ExpandedValidationTracker
from test_run_public_nested_v6 import summary as old_summary, synthetic_roles as small_roles
from test_wandb_training import FakeSDK


@pytest.fixture(autouse=True)
def clean_tracking_environment(monkeypatch):
    for key in tuple(os.environ):
        if key.startswith("WANDB_") or key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
            monkeypatch.delenv(key, raising=False)


def roles_fixture():
    targets = [f"gene_{i:03}" for i in range(56)]
    roster = []
    for source in sorted(runner.SOURCES):
        base, extra = (14, 30) if source == "nadig_jurkat" else (17, 1)
        for index, target in enumerate(targets):
            for batch in range(base + (index < extra)):
                roster.append({"group_id": len(roster), "source": source, "target": target, "batch": f"b{batch}"})
    assert len(roster) == 1767
    roles = {"targets": targets, "excluded_targets": [f"excluded_{i:03}" for i in range(772)],
             "group_roster": roster, "folds": runner.prior.outer_preparation.make_folds(targets, roster)}
    nested = {"inner_folds": {f["fold_id"]: {"fit_targets": f["fit_targets"][:28],
        "tuning_targets": f["fit_targets"][28:]} for f in roles["folds"]},
        "anchor_pools": [{"synthetic_pool": i} for i in range(101)]}
    return roles, nested


def summary(arm, *, roles=None, nested=None, contract=None, sha=None):
    if roles is None:
        # Gate/tracking-only tests do not need expensive expanded group rosters;
        # registration, role/provenance and whole-suite tests use all 1,767.
        roles, nested = small_roles()
    value = old_summary(arm, .98 if arm == "true" else 1., .09 if arm == "true" else .1,
                        roles=roles, nested=nested)
    value.update(schema=runner.SUMMARY_SCHEMA, policy=runner.model.policy())
    if contract is not None:
        value.update(diagnostic_contract_sha256=sha, provenance=copy.deepcopy(runner.provenance(contract, sha)))
    for report in value["folds"]:
        report["training_diagnostics"] = {}
        for candidate in report["inner_selection"]["candidates"]:
            candidate["training_diagnostics"] = {}
    return value


@pytest.fixture
def registration(tmp_path, monkeypatch):
    roles, nested = roles_fixture()
    protocol = tmp_path / "protocol.md"
    protocol.write_text("Synthetic expanded training protocol")
    role_path = tmp_path / "roles.json"
    role_path.write_text("{}")
    role_sha = runner.safe.digest(role_path)
    record = {"protocol": {"path": str(protocol), "sha256": runner.safe.digest(protocol)},
        "inputs": {key: "a" * 64 for key in ("cache_sha256", "contract_sha256", "source_manifest_sha256")},
        "input_paths": {key: str(tmp_path / key) for key in ("cache", "contract", "source_manifest")}}
    monkeypatch.setattr(runner.preparation, "validate_registration", lambda *a, **k: (copy.deepcopy(record), copy.deepcopy(roles), copy.deepcopy(nested)))
    monkeypatch.setattr(runner, "codes", lambda: {"run_public_expanded_v8.py": "b" * 64})
    output = tmp_path / "execution"
    sha = runner.register(role_path, role_sha, protocol, output)
    path = output / "contract.json"
    contract, _, _, _ = runner.validate_contract(path, sha)
    return path, sha, contract, record, roles, nested


def test_native_contract_registration_and_fresh_outputs(registration):
    path, sha, contract, record, roles, nested = registration
    assert contract["schema"] == runner.SCHEMA
    assert contract["expression_decoded"] is False
    assert contract["policy"]["total_events"] == 207
    assert contract["policy"]["gpu_used"] is False
    runner.require_roles(roles, nested)
    with pytest.raises(FileExistsError):
        runner.register(Path(contract["registration"]["path"]), contract["registration"]["sha256"],
                        Path(record["protocol"]["path"]), path.parent)


@pytest.mark.parametrize("part", ["extra", "schema", "bool", "code", "time", "naive", "input", "binding", "protocol"])
def test_rehashed_execution_mutations_fail_closed(registration, part):
    path, _, value, _, _, _ = registration
    if part == "extra": value["ignored"] = True
    elif part == "schema": value["schema"] = runner.prior.SCHEMA
    elif part == "bool": value["expression_decoded"] = 0
    elif part == "code": value["code_sha256"]["run_public_expanded_v8.py"] = "c" * 64
    elif part == "time": value["created_at"] = "invalid"
    elif part == "naive": value["created_at"] = "2026-09-10T12:00:00"
    elif part == "input": value["inputs"]["cache_sha256"] = "c" * 64
    elif part == "binding": value["registration"]["extra"] = 1
    else: value["protocol"]["sha256"] = "c" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        runner.validate_contract(path, runner.safe.digest(path))


@pytest.mark.parametrize("part", ["groups", "pools", "targets", "excluded", "group_role"])
def test_expanded_role_counts_and_direction_identity(part):
    roles, nested = roles_fixture()
    if part == "groups": roles["group_roster"].pop()
    elif part == "pools": nested["anchor_pools"].pop()
    elif part == "targets": roles["targets"].pop()
    elif part == "excluded": roles["excluded_targets"].pop()
    else: roles["folds"][0]["directions"][0]["fit_group_ids"].pop()
    with pytest.raises(ValueError): runner.require_roles(roles, nested)


def test_native_summary_exact_provenance_and_unmodified_gate_projection(registration):
    _, sha, contract, _, roles, nested = registration
    values = {a: summary(a, roles=roles, nested=nested, contract=contract, sha=sha) for a in runner.EVENTS}
    before = copy.deepcopy(values)
    for arm, value in values.items():
        runner.verify_summary(value, arm, sha, contract=contract, roles=roles, nested=nested)
    result = runner.screen(values)
    assert result["passed"] and result["automatic_promotion"] is False
    assert result == runner.prior.screen({a: runner.screen_projection(v) for a, v in values.items()})
    assert values == before
    values["true"]["provenance"]["inputs"]["cache_sha256"] = "c" * 64
    with pytest.raises(ValueError):
        runner.verify_summary(values["true"], "true", sha, contract=contract, roles=roles, nested=nested)


@pytest.mark.parametrize("part", ["schema", "policy", "fold", "target", "source", "lambda", "flag"])
def test_native_summary_mutations_rejected(part):
    roles, nested = roles_fixture()
    value = summary("true", roles=roles, nested=nested)
    if part == "schema": value["schema"] = "public-nested-regularization-summary-v6"
    elif part == "policy": value["policy"] = runner.prior.model.policy()
    elif part == "fold": value["folds"][0]["fit_group_ids"].pop()
    elif part == "target": value["folds"][0]["fit_targets"][0] = value["folds"][0]["held_targets"][0]
    elif part == "source": value["folds"][0]["held_source"] = value["folds"][0]["fit_source"]
    elif part == "lambda": value["folds"][0]["regularization"] = 1.
    else: value["challenge_treated_used"] = 0
    with pytest.raises(ValueError): runner.verify_summary(value, "true", roles=roles, nested=nested)


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("kind", ["conditioning", "distribution", "safety"])
def test_scientific_gates_remain_independent_in_each_source(source, kind):
    values = {a: summary(a) for a in runner.EVENTS}
    metrics = next(r["metrics"] for r in values["true"]["source_metrics"] if r["source"] == source)
    if kind == "conditioning": metrics["target_pooled_mse"] = .991
    elif kind == "distribution": metrics["model_mmd2"] = .101
    else: metrics["negative_fraction"] = .001
    decision = runner.screen(values)
    assert decision[kind + "_passed"] is False
    assert all(decision[k + "_passed"] is True for k in ("conditioning", "distribution", "safety") if k != kind)
    assert decision["passed"] is False


@pytest.mark.parametrize("part", ["old_group_count", "float_count", "bool_audit", "missing", "refit"])
def test_audit_counts_are_new_cohort_and_exact_typed(part):
    audit = complete_audit("true")
    runner.verify_audit(audit, "true")
    if part == "old_group_count": audit["out_of_fold_groups"] = 398
    elif part == "float_count": audit["out_of_fold_groups"] = 1767.
    elif part == "bool_audit": audit["passed"] = 1
    elif part == "missing": audit.pop("inner_selection_replayed")
    else: audit["ridge_refits"] = 1
    with pytest.raises(ValueError): runner.verify_audit(audit, "true")


def setup_preflight(monkeypatch):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    for name in runner.THREAD_VARIABLES: monkeypatch.setenv(name, str(runner.THREADS))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import threadpoolctl
    monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": runner.THREADS}])


@pytest.mark.parametrize("part", ["valid", "head", "gpu", *runner.THREAD_VARIABLES, "active_pool"])
def test_runtime_preflight_rejects_wrong_host_threads_and_gpu(monkeypatch, part):
    setup_preflight(monkeypatch)
    if part == "head": monkeypatch.setattr(runner.socket, "gethostname", lambda: "aida.cac.cornell.edu")
    elif part == "gpu": monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    elif part in runner.THREAD_VARIABLES: monkeypatch.setenv(part, "64")
    elif part == "active_pool":
        import threadpoolctl
        monkeypatch.setattr(threadpoolctl, "threadpool_info", lambda: [{"num_threads": runner.THREADS + 1}])
    if part == "valid": runner.runtime_preflight()
    else:
        with pytest.raises(RuntimeError): runner.runtime_preflight()


@pytest.mark.parametrize("part", ["valid", "rss", "disk", "time", "symlink", "fifo"])
def test_resource_bounds_and_unsafe_output_types(tmp_path, monkeypatch, part):
    monkeypatch.setattr(runner.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=1024))
    monkeypatch.setattr(runner.time, "monotonic", lambda: 10.)
    (tmp_path / "regular").write_bytes(b"abc")
    if part == "rss": monkeypatch.setattr(runner, "MAX_RSS", 1)
    elif part == "disk": monkeypatch.setattr(runner, "MAX_DISK", 2)
    elif part == "time": monkeypatch.setattr(runner, "MAX_SECONDS", 1)
    elif part == "symlink": (tmp_path / "link").symlink_to(tmp_path / "regular")
    elif part == "fifo": os.mkfifo(tmp_path / "fifo")
    if part == "valid":
        result = runner.resources(tmp_path, 0.)
        assert result["output_bytes"] == 3 and result["peak_rss_bytes"] == 1024 * 1024
    else:
        with pytest.raises((RuntimeError, ValueError)): runner.resources(tmp_path, 0.)


def record_tracking(output, value, arm, *, sha="c" * 64, sdk=None):
    tracker = ExpandedValidationTracker(output, arm=arm, config={"diagnostic_contract_sha256": sha}, sdk=sdk or FakeSDK())
    for report in value["folds"]:
        if arm != "control":
            for candidate in report["inner_selection"]["candidates"]:
                tracker.log_inner(report["held_source"], candidate["regularization"], candidate["metrics"], candidate["training_diagnostics"])
        tracker.log_outer(report["held_source"], report["metrics"], report["training_diagnostics"],
                          regularization=None if arm == "control" else report["regularization"])
    tracker.log_outer("all", value["aggregate"])
    tracker.finish(0)
    return output / "tracking.json"


@pytest.mark.parametrize("part", ["valid", "no_metrics", "extra_private", "lambda", "metric", "phase", "order", "bool_index", "receipt", "contract"])
def test_semantic_tracking_replay_rejects_rehashed_journal_changes(tmp_path, part):
    value = summary("true")
    path = record_tracking(tmp_path / "tracking", value, "true")
    sha = "c" * 64
    runner.verify_tracking(path, "true", sha, value)
    journal = path.parent / "metrics.jsonl"
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    if part == "valid":
        assert rows == runner.expected_journal(value, "true")
        return
    if part == "no_metrics": rows = [{"event_index": row["event_index"]} for row in rows]
    elif part == "extra_private": rows[0]["private"] = "not permitted"
    elif part == "lambda":
        key = next(k for k in rows[0] if k.endswith("/regularization"))
        rows[0][key] = 10.
    elif part == "metric":
        key = next(k for k in rows[0] if k.endswith("/target_pooled_mse"))
        rows[0][key] += .0001
    elif part == "phase": rows[0] = {k.replace("inner_regularization", "nested_outer"): v for k, v in rows[0].items()}
    elif part == "order": rows[0], rows[1] = rows[1], rows[0]
    elif part == "bool_index": rows[0]["analysis/event_index"] = True
    elif part == "receipt":
        receipt = json.loads(path.read_text()); receipt["cloud_synced"] = True
        path.write_text(json.dumps(receipt))
    else: sha = "d" * 64
    journal.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError): runner.verify_tracking(path, "true", sha, value)


def complete_audit(arm):
    count = 8 if arm == "control" else 32
    return {**dict.fromkeys(runner.prior.AUDIT_TRUE, True), "ridge_refits": 0,
        "inner_candidates_replayed": 0 if arm == "control" else 24,
        "out_of_fold_groups": 1767, "out_of_fold_source_targets": 112,
        "sharded_artifacts_verified": True, "exact_artifact_roster_verified": True,
        "shard_headers_verified_before_decode": True, "same_open_checksum_verified": True,
        "output_budget_verified": True, "fits_replayed": count, "npz_shards_verified": count * 2,
        "shard_compressed_bytes": 1234, "shard_expanded_bytes": 2345}


def setup_mock_model(registration, monkeypatch, *, error=False):
    path, sha, contract, record, roles, nested = registration
    monkeypatch.setattr(runner, "runtime_preflight", lambda: None)
    monkeypatch.setattr(runner.preparation, "load_feature_tables", lambda _: ({}, {}))
    monkeypatch.setattr(runner.cache, "load_cache", lambda *a, **k: ({"genes": np.asarray(["g"])}, {"excluded_targets": roles["excluded_targets"]}))
    monkeypatch.setattr(runner, "ExpandedValidationTracker", lambda *a, **k: ExpandedValidationTracker(*a, sdk=FakeSDK(), **k))
    def fake_run(arrays, passed_roles, outer_tables, passed_nested, inner_tables, arm, out, provenance, progress=None):
        assert arrays["genes"].flags.writeable is False
        assert passed_roles == roles and passed_nested == nested
        if error: raise RuntimeError("synthetic fit failure")
        value = summary(arm, roles=roles, nested=nested, contract=contract, sha=sha)
        assert provenance == value["provenance"]
        for number, report in enumerate(value["folds"]):
            for candidate in report["inner_selection"]["candidates"]:
                progress({"phase": "inner", "fold_index": number, "held_outer_source": report["held_source"],
                    "regularization": candidate["regularization"], "metrics": candidate["metrics"], "training_diagnostics": {}})
            progress({"phase": "outer", "fold_index": number, "held_outer_source": report["held_source"],
                "regularization": report["regularization"], "metrics": report["metrics"], "training_diagnostics": {}})
        out.mkdir()
        (out / "summary.json").write_text(json.dumps(value, separators=(",", ":")))
        return value
    monkeypatch.setattr(runner.model, "run_arm", fake_run)
    monkeypatch.setattr(runner.model, "audit_arm", lambda out, value, *a: complete_audit(value["arm"]))


def test_mock_suite_orchestration_and_completed_replay_without_fits(registration, monkeypatch, tmp_path):
    setup_mock_model(registration, monkeypatch)
    path, sha, _, _, _, _ = registration
    output = tmp_path / "validation"
    assert runner.run(path, sha, output) == 0
    suite_path = output / "suite.json"
    suite = json.loads(suite_path.read_text())
    assert suite["tracking_events"] == 207 and suite["tracking_errors"] == 0
    assert suite["decision"]["passed"] and suite["next_stage"]["automatic_launch"] is False
    assert suite["next_stage"]["feng_ready"] is suite["next_stage"]["submission_ready"] is False
    assert len({r["run_id"] for r in suite["runs"]}) == 7
    monkeypatch.setattr(runner.model, "run_arm", lambda *a, **k: pytest.fail("Completed verification fitted a model"))
    monkeypatch.setattr(runner, "ExpandedValidationTracker", lambda *a, **k: pytest.fail("Verification initialized W&B"))
    audit = runner.verify_suite(suite_path, runner.safe.digest(suite_path))
    assert audit["verified"] and audit["ridge_refits"] == 0 and audit["tracking_events_replayed"] == 207
    complete_path = output / "complete.json"
    complete = json.loads(complete_path.read_text())
    for part in ("extra", "extra_run", "runtime_extra", "runtime_bool", "runtime_host", "runtime_gpu",
                 "runtime_threads", "runtime_memory", "runtime_disk", "runtime_time", "completion_time",
                 "run_path", "run_audit", "run_order", "duplicate_run", "readiness"):
        changed = copy.deepcopy(suite)
        if part == "extra": changed["ignored"] = True
        elif part == "extra_run": changed["runs"][0]["ignored"] = True
        elif part == "runtime_extra": changed["runtime"]["ignored"] = True
        elif part == "runtime_bool": changed["runtime"]["peak_rss_bytes"] = True
        elif part == "runtime_host": changed["runtime"]["hostname"] = "login.anvil.rcac.purdue.edu"
        elif part == "runtime_gpu": changed["runtime"]["device"] = "cuda"
        elif part == "runtime_threads": changed["runtime"]["cpu_threads"] = 16
        elif part == "runtime_memory": changed["runtime"]["peak_rss_bytes"] = runner.MAX_RSS + 1
        elif part == "runtime_disk": changed["runtime"]["output_bytes"] = runner.MAX_DISK + 1
        elif part == "runtime_time": changed["runtime"]["elapsed_seconds"] = runner.MAX_SECONDS + 1
        elif part == "completion_time": changed["completed_at"] = "2000-01-01T00:00:00+00:00"
        elif part == "run_path": changed["runs"][0]["summary_path"] = str(tmp_path / "elsewhere.json")
        elif part == "run_audit": changed["runs"][0]["artifact_audit"]["sharded_artifacts_verified"] = False
        elif part == "run_order": changed["runs"][0], changed["runs"][1] = changed["runs"][1], changed["runs"][0]
        elif part == "duplicate_run": changed["runs"][1]["run_id"] = changed["runs"][0]["run_id"]
        else: changed["next_stage"]["submission_ready"] = True
        suite_path.write_text(json.dumps(changed))
        completion = {**complete, "suite_sha256": runner.safe.digest(suite_path), "completed_at": changed["completed_at"]}
        complete_path.write_text(json.dumps(completion))
        with pytest.raises(ValueError): runner.verify_suite(suite_path, runner.safe.digest(suite_path))
    suite_path.write_text(json.dumps(suite))
    complete_path.write_text(json.dumps({**complete, "suite_sha256": runner.safe.digest(suite_path)}))


def test_failed_mock_model_preserves_tracking_and_no_completion(registration, monkeypatch, tmp_path):
    setup_mock_model(registration, monkeypatch, error=True)
    path, sha, _, _, _, _ = registration
    output = tmp_path / "failed-validation"
    with pytest.raises(RuntimeError, match="synthetic fit failure"): runner.run(path, sha, output)
    assert not (output / "complete.json").exists() and not (output / "suite.json").exists()
    receipt = json.loads((output / "tracking/control/tracking.json").read_text())
    assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 1
    assert receipt["event_count"] == 0
