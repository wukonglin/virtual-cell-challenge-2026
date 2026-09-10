"""Synthetic sparse-runner admission, separate screens and offline orchestration."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_target_context_v5 as runner
from public_target_context_tracking import TargetContextTracker
from test_wandb_training import FakeSDK


def summary(arm, primary=1.0, mmd=.1):
    metrics = dict.fromkeys(runner.METRICS, .1)
    metrics.update(target_pooled_mse=primary, model_mmd2=mmd, negative_fraction=0.,
        zero_fraction=.6, control_zero_fraction=.6, treated_zero_fraction=.6,
        unchanged_count_fraction=1., occupancy_clip_fraction=0., amplitude_clip_fraction=0.,
        unresolved_activation_fraction=0., requested_realized_occupancy_mse=0.)
    return {"schema": "public-target-context-summary-v5", "arm": arm,
        "status": "completed_unpromoted_diagnostic", "stage": "validation",
        "policy": runner.model.policy(), "prior_public_development_exposure_acknowledged": True,
        "aggregate": dict(metrics),
        "source_metrics": [{"source": source, "metrics": dict(metrics)} for source in sorted(runner.SOURCES)],
        "folds": [{"fold_id": str(i), "fit_targets": [str(t) for t in range(42)],
            "held_targets": [str(t) for t in range(42, 56)], "metrics": dict(metrics)} for i in range(8)],
        "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True,
        **dict.fromkeys(runner.FLAGS_FALSE, False)}


def candidates():
    return {arm: summary(arm, .98 if arm == "true" else 1., .09 if arm == "true" else .1)
            for arm in runner.model.ARMS}


def metric(values, arm, source):
    return next(row["metrics"] for row in values[arm]["source_metrics"] if row["source"] == source)


def test_separate_screens_pass_without_any_promotion_claim():
    decision = runner.screen(candidates())
    assert all(decision[key] for key in ("passed", "conditioning_passed", "distribution_passed", "safety_passed"))
    assert decision["automatic_promotion"] is False and decision["leaderboard_improvement_claimed"] is False
    assert {row["source"] for row in decision["sources"]} == runner.SOURCES


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", *runner.model.SHUFFLED_ARMS))
def test_each_source_and_primary_comparator_requires_one_percent(source, arm):
    values = candidates()
    metric(values, arm, source)["target_pooled_mse"] = .989  # Gain below one percent.
    decision = runner.screen(values)
    assert not decision["conditioning_passed"] and not decision["passed"]
    assert decision["distribution_passed"] and decision["safety_passed"]
    failed = next(row for row in decision["sources"] if row["source"] == source)
    assert failed["checks"][f"primary_vs_{arm}"] is False


def test_exact_one_percent_boundary_is_inclusive_despite_ratio_roundoff():
    values = candidates()
    for source in runner.SOURCES:
        for arm in runner.model.ARMS:
            metric(values, arm, source)["target_pooled_mse"] = .99 * 33. if arm == "true" else 33.
    assert runner.screen(values)["conditioning_passed"] is True


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
def test_context_primary_requires_strict_improvement_not_tie(source):
    values = candidates()
    metric(values, "context", source)["target_pooled_mse"] = .98
    decision = runner.screen(values)
    assert not decision["conditioning_passed"] and decision["distribution_passed"] and decision["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", "context", "additive", *runner.model.SHUFFLED_ARMS))
def test_mmd_noninferiority_against_each_comparator_and_source(source, arm):
    values = candidates()
    metric(values, arm, source)["model_mmd2"] = .08
    decision = runner.screen(values)
    assert not decision["distribution_passed"] and decision["conditioning_passed"] and decision["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", "context"))
def test_occupancy_mse_noninferior_to_control_and_context(source, arm):
    values = candidates()
    metric(values, arm, source)["target_pooled_occupancy_mse"] = .09
    decision = runner.screen(values)
    assert not decision["distribution_passed"] and decision["conditioning_passed"] and decision["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
def test_aggregate_zero_rate_error_is_absolute_and_source_specific(source):
    values = candidates()
    row = metric(values, "true", source)
    row.update(treated_zero_fraction=.5, control_zero_fraction=.6, zero_fraction=.39)
    decision = runner.screen(values)
    assert not decision["distribution_passed"] and decision["conditioning_passed"] and decision["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("key,bad", [("negative_fraction", .000001), ("unresolved_activation_fraction", .000001),
    ("occupancy_clip_fraction", .010001), ("amplitude_clip_fraction", .010001)])
def test_safety_failures_do_not_masquerade_as_conditioning_failures(source, key, bad):
    values = candidates()
    metric(values, "true", source)[key] = bad
    decision = runner.screen(values)
    assert not decision["safety_passed"] and not decision["passed"]
    assert decision["conditioning_passed"] and decision["distribution_passed"]


@pytest.mark.parametrize("key", ("occupancy_clip_fraction", "amplitude_clip_fraction"))
def test_one_percent_clip_boundary_is_inclusive(key):
    values = candidates()
    for source in runner.SOURCES:
        metric(values, "true", source)[key] = .01
    assert runner.screen(values)["safety_passed"]


def test_zero_primary_comparator_denominator_does_not_pass():
    values = candidates()
    source = sorted(runner.SOURCES)[0]
    metric(values, "control", source)["target_pooled_mse"] = 0.
    decision = runner.screen(values)
    assert not decision["passed"]
    assert next(r for r in decision["sources"] if r["source"] == source)["relative_primary_improvement"]["control"] is None


@pytest.mark.parametrize("arm", runner.model.ARMS)
def test_every_registered_arm_required(arm):
    values = candidates()
    values.pop(arm)
    with pytest.raises(ValueError, match="seven"):
        runner.screen(values)


@pytest.mark.parametrize("key", runner.FLAGS_FALSE)
@pytest.mark.parametrize("bad", [True, None, 0])
def test_activity_flags_must_be_explicit_false(key, bad):
    value = summary("true")
    value[key] = bad
    with pytest.raises(ValueError, match="Forbidden"):
        runner.verify_summary(value, "true")


@pytest.mark.parametrize("field", ["schema", "status", "stage", "arm", "policy", "prior_public_development_exposure_acknowledged",
    "exact_ntc_identity_verified", "exact_zero_effect_identity_verified"])
def test_summary_identity_schema_and_policy_fail_closed(field):
    value = summary("true")
    value.pop(field)
    with pytest.raises(ValueError):
        runner.verify_summary(value, "true")


@pytest.mark.parametrize("part", ["missing_metric", "nan", "negative_mse", "fraction", "boolean", "sources", "duplicate_source",
    "folds", "duplicate_fold", "fit_count", "held_count", "contract", "nested_contract"])
def test_summary_metrics_roster_and_provenance_fail_closed(part):
    value = summary("true")
    value["diagnostic_contract_sha256"] = "a" * 64
    value["provenance"] = {"diagnostic_contract_sha256": "a" * 64}
    if part == "missing_metric": value["aggregate"].pop(runner.METRICS[-1])
    elif part == "nan": value["aggregate"]["model_mmd2"] = float("nan")
    elif part == "negative_mse": value["aggregate"]["target_pooled_mse"] = -.01
    elif part == "fraction": value["aggregate"]["zero_fraction"] = 1.01
    elif part == "boolean": value["aggregate"]["model_centroid_mse"] = True
    elif part == "sources": value["source_metrics"].pop()
    elif part == "duplicate_source": value["source_metrics"][1]["source"] = value["source_metrics"][0]["source"]
    elif part == "folds": value["folds"].pop()
    elif part == "duplicate_fold": value["folds"][1]["fold_id"] = value["folds"][0]["fold_id"]
    elif part == "fit_count": value["folds"][0]["fit_targets"].pop()
    elif part == "held_count": value["folds"][0]["held_targets"].pop()
    elif part == "contract": value["diagnostic_contract_sha256"] = "b" * 64
    else: value["provenance"]["diagnostic_contract_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        runner.verify_summary(value, "true", "a" * 64)


@pytest.fixture
def registered(tmp_path, monkeypatch):
    parent, role, protocol = (tmp_path / name for name in ("parent.json", "roles.json", "protocol.md"))
    parent.write_text("Synthetic opaque v3 parent; not an expression archive")
    role.write_text("Synthetic opaque role metadata")
    protocol.write_text("Synthetic frozen sparse output protocol")
    parent_sha, role_sha = runner.sha256_file(parent), runner.sha256_file(role)
    monkeypatch.setattr(runner, "PRIOR_SHA", parent_sha)
    roles = {"inputs": {"cache_sha256": "a" * 64, "contract_sha256": "b" * 64,
        "source_manifest_sha256": "c" * 64},
        "input_paths": {key: str(tmp_path / key) for key in ("cache", "contract", "source_manifest")},
        "excluded_targets": [str(i) for i in range(772)]}
    previous = {"registration": {"path": str(role), "sha256": role_sha}}
    def validate(path, digest):
        runner.authenticated_bytes(path, digest, 1 << 20)
        runner.authenticated_bytes(role, role_sha, 1 << 20)
        return copy.deepcopy(previous), copy.deepcopy(roles)
    monkeypatch.setattr(runner.prior, "validate_contract", validate)
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("No expression decoding in registration"))
    monkeypatch.setattr(runner.preparation, "load_feature_tables", lambda *a, **k: pytest.fail("No feature decoding in registration"))
    output = tmp_path / "registration"
    digest = runner.register(parent, protocol, output)
    return {"path": output / "contract.json", "digest": digest, "parent": parent, "role": role,
            "protocol": protocol, "roles": roles, "previous": previous}


def test_registration_is_expression_free_and_pins_all_inputs(registered):
    value, roles = runner.validate_contract(registered["path"], registered["digest"])
    assert value["expression_decoded"] is False and value["cpu_only"] is True
    assert value["wandb_mode"] == "offline" and value["events_per_arm"] == 9
    assert roles == registered["roles"]
    assert set(value["code_sha256"]) == set(runner.MODULES)
    assert value["protocol"]["sha256"] == runner.sha256_file(registered["protocol"])


@pytest.mark.parametrize("field,bad", [("schema", "changed"), ("automatic_promotion", True), ("submission_allowed", True),
    ("expression_decoded", True), ("new_raw_expression_allowed", True), ("cpu_only", False), ("stage", "pretrain"),
    ("wandb_mode", "online"), ("events_per_arm", 8), ("policy", {}), ("screen", {})])
def test_registration_policy_cannot_be_relaxed(registered, field, bad):
    path = registered["path"]
    record = json.loads(path.read_text())
    record[field] = bad
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="policy"):
        runner.validate_contract(path, runner.sha256_file(path))


@pytest.mark.parametrize("which", ["parent", "role", "protocol"])
def test_changed_parent_role_or_protocol_bytes_rejected(registered, which):
    registered[which].write_text("changed metadata")
    with pytest.raises(ValueError, match="SHA"):
        runner.validate_contract(registered["path"], registered["digest"])


@pytest.mark.parametrize("part", ["code", "missing_code", "parent_pin", "input", "registration"])
def test_changed_code_or_parent_bindings_fail_closed(registered, part):
    path = registered["path"]
    record = json.loads(path.read_text())
    if part == "code": record["code_sha256"][runner.MODULES[0]] = "f" * 64
    elif part == "missing_code": record["code_sha256"].pop(runner.MODULES[0])
    elif part == "parent_pin": record["prior_v3_contract"]["sha256"] = "f" * 64
    elif part == "input": record["inputs"]["cache_sha256"] = "f" * 64
    else: record["registration"]["sha256"] = "f" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runner.validate_contract(path, runner.sha256_file(path))


def test_existing_registration_never_overwritten(registered):
    with pytest.raises(FileExistsError):
        runner.register(registered["parent"], registered["protocol"], registered["path"].parent)


@pytest.mark.parametrize("host", ["login02.anvil", "aida.cac.cornell.edu", "cbsulogin.biohpc.cornell.edu", "cbsuvlaminck3-pretend"])
def test_fake_slurm_does_not_authorize_login_or_lookalike_host(tmp_path, monkeypatch, host):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setenv("SLURM_JOB_ID", "20539599")
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("No decoding on login"))
    with pytest.raises(RuntimeError, match="compute host"):
        runner.run(tmp_path / "none", "a" * 64, tmp_path / "run")
    assert not (tmp_path / "run").exists()


def install_synthetic_run(registered, monkeypatch, failure=None):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setattr(runner.preparation, "load_feature_tables", lambda *a: {"synthetic": True})
    arrays = {"synthetic": np.zeros(2)}
    row_contract = copy.deepcopy(registered["roles"])
    if failure == "exclusions": row_contract["excluded_targets"].pop()
    monkeypatch.setattr(runner, "load_cache", lambda *a: (arrays, row_contract))
    monkeypatch.setattr(runner.model, "verify_artifacts", lambda *a: {"passed": failure != "artifact_audit", "synthetic": True})
    sdks = []
    def tracker(*args, **kwargs):
        sdk = FakeSDK(log_error=failure == "log", finish_error=failure == "finish")
        sdks.append(sdk)
        result = TargetContextTracker(*args, sdk=sdk, **kwargs)
        if failure == "duplicate_run_ids": result.record["run_id"] = "same_synthetic_id"
        return result
    monkeypatch.setattr(runner, "TargetContextTracker", tracker)
    def fit(arrays, roles, tables, arm, output_dir, provenance, progress=None):
        assert not arrays["synthetic"].flags.writeable
        assert roles == registered["roles"] and tables == {"synthetic": True}
        if failure == "model": raise RuntimeError("Synthetic model failure")
        output_dir.mkdir()
        value = candidates()[arm]
        value["diagnostic_contract_sha256"] = provenance["diagnostic_contract_sha256"]
        value["provenance"] = provenance
        for i in range(7 if failure == "events" else 8):
            progress({"fold_index": i, "held_source": sorted(runner.SOURCES)[i % 2], "metrics": value["aggregate"], "training_diagnostics": {}})
        if failure == "provenance": value["provenance"] = {"diagnostic_contract_sha256": "f" * 64}
        if failure == "final_protocol" and arm == runner.model.ARMS[-1]: registered["protocol"].write_text("changed during execution")
        runner.dump_new(output_dir / "summary.json", value)
        return value
    monkeypatch.setattr(runner.model, "run_arm", fit)
    return sdks


def test_mock_seven_arm_run_has_63_events_and_identical_completion_timestamps(registered, tmp_path, monkeypatch):
    sdks = install_synthetic_run(registered, monkeypatch)
    output = tmp_path / "run"
    assert runner.run(registered["path"], registered["digest"], output) == 0
    suite = json.loads((output / "suite.json").read_text())
    complete = json.loads((output / "complete.json").read_text())
    assert suite["completed_at"] == complete["completed_at"]
    assert complete["suite_sha256"] == runner.sha256_file(output / "suite.json")
    assert complete["screen_passed"] is True
    assert suite["tracking_events"] == complete["tracking_events"] == 63
    assert suite["tracking_errors"] == complete["tracking_errors"] == 0
    assert suite["promoted"] is complete["promoted"] is False
    assert suite["submission_performed"] is complete["submission_performed"] is False
    assert suite["cloud_synced"] is complete["cloud_synced"] is False
    assert all(suite[k] is False for k in ("pretraining_performed", "posttraining_performed", "new_raw_expression_read", "gpu_used", "download_performed"))
    assert len(sdks) == len(suite["runs"]) == 7
    for sdk in sdks:
        assert len(sdk.run.logged) == 9
        assert sdk.initialized[0]["mode"] == "offline"
        assert [data["analysis/event_index"] for data, _ in sdk.run.logged] == list(range(1, 10))
    with pytest.raises(FileExistsError):
        runner.run(registered["path"], registered["digest"], output)


@pytest.mark.parametrize("failure", ["model", "log", "finish", "events", "provenance", "artifact_audit", "exclusions",
                                     "duplicate_run_ids", "final_protocol"])
def test_failure_never_leaves_completed_suite(registered, tmp_path, monkeypatch, failure):
    sdks = install_synthetic_run(registered, monkeypatch, failure)
    output = tmp_path / "run"
    with pytest.raises((ValueError, RuntimeError)):
        runner.run(registered["path"], registered["digest"], output)
    assert not (output / "complete.json").exists()
    assert not (output / "suite.json").exists()
    if failure in {"model", "provenance", "artifact_audit"}:
        receipt = json.loads((output / "tracking/control/tracking.json").read_text())
        assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 1
    if failure == "exclusions": assert not sdks

@pytest.mark.parametrize("source", sorted(runner.SOURCES))
def test_additive_primary_requires_strict_improvement_not_tie(source):
    values = candidates()
    metric(values, "additive", source)["target_pooled_mse"] = .98
    assert runner.screen(values)["conditioning_passed"] is False

@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ["control", "context"])
def test_batch_centroid_cannot_worsen_despite_pooled_gain(source, arm):
    values = candidates()
    metric(values, arm, source)["model_centroid_mse"] = .09
    assert runner.screen(values)["conditioning_passed"] is True
    assert runner.screen(values)["distribution_passed"] is False
