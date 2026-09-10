"""Synthetic nested-runner gates, exact role/provenance binding and offline runs."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_public_nested_v6 as runner
from public_nested_tracking_v6 import NestedValidationTracker
from test_wandb_training import FakeSDK


def synthetic_roles():
    targets = [f"gene_{i:03}" for i in range(56)]
    roster = []
    for source in sorted(runner.SOURCES):
        extra = 23 if source == "nadig_jurkat" else 39
        for i, target in enumerate(targets):
            for batch in range(4 if i < extra else 3):
                roster.append({"group_id": len(roster), "source": source, "target": target, "batch": f"b{batch}"})
    assert len(roster) == 398
    roles = {"targets": targets, "group_roster": roster,
             "excluded_targets": [f"excluded_{i:03}" for i in range(772)],
             "folds": runner.outer_preparation.make_folds(targets, roster)}
    nested = {"inner_folds": {fold["fold_id"]: {"fit_targets": fold["fit_targets"][:28],
                 "tuning_targets": fold["fit_targets"][28:]} for fold in roles["folds"]}}
    return roles, nested


def scalar_metrics(primary=1., mmd=.1):
    result = dict.fromkeys(runner.METRICS, .1)
    result.update(target_pooled_mse=primary, model_mmd2=mmd, negative_fraction=0.,
        zero_fraction=.6, control_zero_fraction=.6, treated_zero_fraction=.6,
        unchanged_count_fraction=1., occupancy_clip_fraction=0., amplitude_clip_fraction=0.,
        unresolved_activation_fraction=0., requested_realized_occupancy_mse=0.,
        raw_model_mmd2=mmd, raw_control_mmd2=.1)
    return result


def fold_report(fold, metrics, roster, *, prefix, kind, parent="", lam=10.):
    return {"fold_id": fold.fold_id, "target_fold": fold.target_fold,
        "fit_source": fold.fit_source, "held_source": fold.held_source,
        "fit_targets": list(fold.fit_targets), "held_targets": list(fold.held_targets),
        "fit_group_ids": list(fold.fit_indices), "joint_eval_group_ids": list(fold.evaluation_indices),
        "held_target_group_ids_all_sources": list(fold.held_target_indices_all_sources),
        "groups": [{**{k: roster[g][k] for k in ("source", "target", "batch")},
                    "group": g, **metrics} for g in fold.evaluation_indices],
        "metrics": dict(metrics), "validation_kind": kind, "outer_fold_id": parent,
        "regularization": lam, "array_prefix": prefix}


def summary(arm, primary=1., mmd=.1, *, roles=None, nested=None):
    if roles is None: roles, nested = synthetic_roles()
    axes, folds = runner.expected_folds(roles)
    metrics = scalar_metrics(primary, mmd)
    reports = []
    for number, fold in enumerate(folds):
        outer = fold_report(fold, metrics, roles["group_roster"], prefix=f"fold_{number}_outer_",
                            kind="outer_joint_target_source")
        candidates = []
        if arm != "control":
            inner = runner.model.inner_fold(axes, fold, nested)
            candidates = [fold_report(inner, scalar_metrics(.7), roles["group_roster"],
                prefix=f"fold_{number}_inner_{index}_", kind="inner_same_source_target", parent=fold.fold_id, lam=lam)
                for index, lam in enumerate(runner.model.GRID)]
        outer["inner_selection"] = runner.model.selection_record(arm, candidates, 10.)
        reports.append(outer)
    return {"schema": "public-nested-regularization-summary-v6", "arm": arm,
        "status": "completed_unpromoted_diagnostic", "stage": "validation", "policy": runner.model.policy(),
        "prior_public_development_exposure_acknowledged": True,
        "exact_ntc_identity_verified": True, "exact_zero_effect_identity_verified": True,
        "hyperparameter_selection": arm != "control", "inner_selection_performed": arm != "control",
        "selection_scope": "inner_only_disjoint_reference_rows", "aggregate": dict(metrics),
        "source_metrics": [{"source": source, "metrics": dict(metrics)} for source in sorted(runner.SOURCES)],
        "folds": reports, "runtime": {"hostname": "cbsuvlaminck3.biohpc.cornell.edu", "device": "cpu"},
        **dict.fromkeys(runner.FLAGS_FALSE, False)}


def candidates():
    roles, nested = synthetic_roles()
    return {arm: summary(arm, .98 if arm == "true" else 1., .09 if arm == "true" else .1,
                         roles=roles, nested=nested) for arm in runner.model.ARMS}


def metric(values, arm, source):
    return next(r["metrics"] for r in values[arm]["source_metrics"] if r["source"] == source)


def test_exact_role_counts_and_separate_screens_have_no_promotion_claim():
    roles, nested = synthetic_roles()
    for arm in runner.model.ARMS:
        value = summary(arm, roles=roles, nested=nested)
        runner.verify_summary(value, arm, roles=roles, nested=nested)
        assert len(value["folds"]) == 8
        assert sum(len(f["inner_selection"]["candidates"]) for f in value["folds"]) == (0 if arm == "control" else 24)
    decision = runner.screen(candidates())
    assert all(decision[k] for k in ("passed", "conditioning_passed", "distribution_passed", "safety_passed"))
    assert decision["automatic_promotion"] is decision["leaderboard_improvement_claimed"] is False
    assert all(len(row["checks"]) == 22 for row in decision["sources"])


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", *runner.model.SHUFFLED_ARMS))
def test_one_percent_primary_required_for_each_source_and_comparator(source, arm):
    values = candidates(); metric(values, arm, source)["target_pooled_mse"] = .989
    result = runner.screen(values)
    assert not result["conditioning_passed"] and result["distribution_passed"] and result["safety_passed"]


def test_primary_boundary_is_inclusive_without_relaxing_inequality():
    values = candidates()
    for source in runner.SOURCES:
        for arm in runner.model.ARMS:
            metric(values, arm, source)["target_pooled_mse"] = .99 * 33. if arm == "true" else 33.
    assert runner.screen(values)["conditioning_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("context", "additive"))
def test_context_and_additive_require_strict_primary_improvement(source, arm):
    values = candidates(); metric(values, arm, source)["target_pooled_mse"] = .98
    result = runner.screen(values)
    assert not result["conditioning_passed"] and result["distribution_passed"] and result["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", "context", "additive", *runner.model.SHUFFLED_ARMS))
def test_mmd_noninferiority_against_every_comparator(source, arm):
    values = candidates(); metric(values, arm, source)["model_mmd2"] = .08
    result = runner.screen(values)
    assert result["conditioning_passed"] and not result["distribution_passed"] and result["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("arm", ("control", "context"))
@pytest.mark.parametrize("key", ("model_centroid_mse", "target_pooled_occupancy_mse"))
def test_secondary_occupancy_and_batch_mse_cannot_worsen(source, arm, key):
    values = candidates(); metric(values, arm, source)[key] = .09
    result = runner.screen(values)
    assert result["conditioning_passed"] and not result["distribution_passed"] and result["safety_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
def test_zero_rate_absolute_error_is_source_specific(source):
    values = candidates(); metric(values, "true", source).update(zero_fraction=.39, treated_zero_fraction=.5,
                                                                control_zero_fraction=.6)
    assert not runner.screen(values)["distribution_passed"]


@pytest.mark.parametrize("source", sorted(runner.SOURCES))
@pytest.mark.parametrize("key,bad", [("negative_fraction", 1e-9), ("unresolved_activation_fraction", 1e-9),
                                  ("occupancy_clip_fraction", .010001), ("amplitude_clip_fraction", .010001)])
def test_safety_is_distinct_from_conditioning_and_distribution(source, key, bad):
    values = candidates(); metric(values, "true", source)[key] = bad
    result = runner.screen(values)
    assert result["conditioning_passed"] and result["distribution_passed"] and not result["safety_passed"]


@pytest.mark.parametrize("key", ("occupancy_clip_fraction", "amplitude_clip_fraction"))
def test_clip_bound_is_exactly_inclusive(key):
    values = candidates()
    for source in runner.SOURCES: metric(values, "true", source)[key] = .01
    assert runner.screen(values)["safety_passed"]


def test_zero_comparator_primary_denominator_cannot_pass():
    values = candidates(); source = sorted(runner.SOURCES)[0]
    metric(values, "control", source)["target_pooled_mse"] = 0.
    result = runner.screen(values)
    assert not result["conditioning_passed"]
    assert next(row for row in result["sources"] if row["source"] == source)["relative_primary_improvement"]["control"] is None


@pytest.mark.parametrize("arm", runner.model.ARMS)
def test_missing_any_arm_rejects_screen(arm):
    values = candidates(); del values[arm]
    with pytest.raises(ValueError): runner.screen(values)


@pytest.mark.parametrize("key", runner.FLAGS_FALSE)
@pytest.mark.parametrize("bad", [True, None, 0])
def test_explicit_false_activity_flags_reject_bool_lookalikes(key, bad):
    value = summary("true"); value[key] = bad
    with pytest.raises(ValueError): runner.verify_summary(value, "true")


@pytest.mark.parametrize("key", ["schema", "arm", "status", "stage", "policy", "runtime",
    "prior_public_development_exposure_acknowledged", "exact_ntc_identity_verified", "exact_zero_effect_identity_verified",
    "hyperparameter_selection", "inner_selection_performed", "selection_scope"])
def test_summary_schema_identity_and_policy_fail_closed(key):
    value = summary("true"); value.pop(key)
    with pytest.raises(ValueError): runner.verify_summary(value, "true")


@pytest.mark.parametrize("value", [True, False, None, "0", float("nan"), float("inf"), -1e-9, 10**1000])
def test_invalid_canonical_metric_types_and_values_rejected(value):
    metrics = scalar_metrics(); metrics["model_mmd2"] = value
    with pytest.raises(ValueError): runner.verify_metrics(metrics)


@pytest.mark.parametrize("key", ("raw_model_mmd2", "raw_control_mmd2"))
@pytest.mark.parametrize("value", [-1e-12, -1e-14, 0., .1])
def test_tiny_signed_raw_mmd_allowed_without_changing_canonical_value(key, value):
    metrics = scalar_metrics(); metrics[key] = value; metrics["model_mmd2"] = 0.
    runner.verify_metrics(metrics)
    assert metrics[key] == value and metrics["model_mmd2"] == 0.


@pytest.mark.parametrize("value", [-1.000001e-12, True, None, float("nan"), float("inf"), 10**1000])
def test_invalid_raw_mmd_rejected(value):
    metrics = scalar_metrics(); metrics["raw_model_mmd2"] = value
    with pytest.raises(ValueError): runner.verify_metrics(metrics)


@pytest.mark.parametrize("part", ["missing_metric", "fraction", "duplicate_source", "missing_source", "duplicate_fold",
    "missing_fold", "duplicate_outer_target", "duplicate_inner_target", "overlap_outer_target", "inner_fit_count",
    "inner_held_count", "outer_fit_count", "outer_held_count", "inner_source", "outer_source", "outer_prefix",
    "inner_prefix", "inner_parent", "outer_parent", "integer_lambda", "boolean_lambda", "selected_lambda"])
def test_nested_roles_counts_prefixes_and_selected_lambda_fail_closed(part):
    value = summary("true"); outer = value["folds"][0]; inner = outer["inner_selection"]["candidates"][0]
    if part == "missing_metric": value["aggregate"].pop("model_mmd2")
    elif part == "fraction": value["aggregate"]["zero_fraction"] = 1.01
    elif part == "duplicate_source": value["source_metrics"][1]["source"] = value["source_metrics"][0]["source"]
    elif part == "missing_source": value["source_metrics"].pop()
    elif part == "duplicate_fold": value["folds"][1]["fold_id"] = outer["fold_id"]
    elif part == "missing_fold": value["folds"].pop()
    elif part == "duplicate_outer_target": outer["fit_targets"][1] = outer["fit_targets"][0]
    elif part == "duplicate_inner_target": inner["fit_targets"][1] = inner["fit_targets"][0]
    elif part == "overlap_outer_target": outer["held_targets"] = sorted([outer["fit_targets"][0], *outer["held_targets"][1:]])
    elif part == "inner_fit_count": inner["fit_targets"].pop()
    elif part == "inner_held_count": inner["held_targets"].pop()
    elif part == "outer_fit_count": outer["fit_targets"].pop()
    elif part == "outer_held_count": outer["held_targets"].pop()
    elif part == "inner_source": inner["held_source"] = outer["held_source"]
    elif part == "outer_source": outer["held_source"] = outer["fit_source"]
    elif part == "outer_prefix": outer["array_prefix"] = "fold_9_outer_"
    elif part == "inner_prefix": inner["array_prefix"] = "fold_0_inner_9_"
    elif part == "inner_parent": inner["outer_fold_id"] = "wrong"
    elif part == "outer_parent": outer["outer_fold_id"] = "wrong"
    elif part == "integer_lambda": inner["regularization"] = 1
    elif part == "boolean_lambda": inner["regularization"] = True
    else: outer["regularization"] = .1
    with pytest.raises(ValueError): runner.verify_summary(value, "true")


@pytest.mark.parametrize("scores,expected", [([.3, .3, .3], 10.), ([.3, .2, .3], 1.),
    ([.2, .3, .3], .1), ([.3, .3, .2], 10.), ([.3, .3-1e-14, .3], 1.)])
def test_candidate_selection_exact_ties_choose_stronger_without_score_tolerance(scores, expected):
    value = summary("true"); fold = value["folds"][0]; choices = fold["inner_selection"]["candidates"]
    for candidate, score in zip(choices, scores): candidate["metrics"]["target_pooled_mse"] = score
    assert runner.model.select_regularization(choices) == expected
    fold["inner_selection"] = runner.model.selection_record("true", choices, expected)
    fold["regularization"] = expected
    runner.verify_summary(value, "true")


@pytest.mark.parametrize("part", ["missing", "duplicate", "order", "score_bool", "score_nan", "score_negative", "score_huge"])
def test_bad_candidate_grid_and_scores_rejected(part):
    value = summary("true"); choices = value["folds"][0]["inner_selection"]["candidates"]
    if part == "missing": choices.pop()
    elif part == "duplicate": choices[1]["regularization"] = .1
    elif part == "order": choices.reverse()
    else: choices[0]["metrics"]["target_pooled_mse"] = {
        "score_bool": True, "score_nan": float("nan"), "score_negative": -.1, "score_huge": 10**1000}[part]
    with pytest.raises(ValueError): runner.verify_summary(value, "true")


def test_outer_score_never_selects_lambda_and_control_has_no_candidates():
    value = summary("true"); value["folds"][0]["metrics"]["target_pooled_mse"] = 999.
    runner.verify_summary(value, "true")
    control = summary("control"); control["folds"][0]["inner_selection"]["candidates"] = \
        value["folds"][0]["inner_selection"]["candidates"]
    with pytest.raises(ValueError): runner.verify_summary(control, "control")


@pytest.fixture
def registered(tmp_path, monkeypatch):
    parent, role, protocol = (tmp_path / name for name in ("outer.json", "nested.json", "protocol.md"))
    parent.write_text("Synthetic opaque outer role manifest")
    role.write_text("Synthetic opaque nested registration")
    protocol.write_text("Synthetic frozen nested protocol")
    real_digest = runner.sha256_file
    def digest(path):
        path = Path(path)
        if path.parent == Path(runner.__file__).parent:
            return hashlib.sha256(("synthetic-code:" + path.name).encode()).hexdigest()
        return real_digest(path)
    monkeypatch.setattr(runner, "sha256_file", digest)
    parent_sha, role_sha, protocol_sha = map(digest, (parent, role, protocol))
    roles, nested = synthetic_roles()
    roles.update(inputs={"cache_sha256": "a"*64, "contract_sha256": "b"*64, "source_manifest_sha256": "c"*64},
        input_paths={key: str(tmp_path/key) for key in ("cache", "contract", "source_manifest")})
    nested.update(outer_registration={"path": str(parent), "sha256": parent_sha},
                  protocol={"path": str(protocol), "sha256": protocol_sha})
    def validate(path, sha):
        if Path(path) != role or sha != role_sha or digest(path) != role_sha or digest(parent) != parent_sha:
            raise ValueError("Synthetic registration SHA binding changed")
        return copy.deepcopy(nested), copy.deepcopy(roles)
    monkeypatch.setattr(runner.preparation, "validate_registration", validate)
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("Registration must not decode expressions"))
    for module in (runner.preparation, runner.outer_preparation):
        monkeypatch.setattr(module, "load_feature_tables", lambda *a, **k: pytest.fail("Registration must not decode feature tables"))
    output = tmp_path/"registration"
    sha = runner.register(role, role_sha, protocol, output)
    return {"path": output/"contract.json", "digest": sha, "role": role, "role_sha": role_sha,
            "parent": parent, "protocol": protocol, "roles": roles, "nested": nested}


def test_expression_free_registration_pins_inputs_code_policy_and_event_counts(registered):
    contract, roles, nested = runner.validate_contract(registered["path"], registered["digest"])
    assert roles == registered["roles"] and nested == registered["nested"]
    assert contract["expression_decoded"] is False and contract["cpu_only"] is True
    assert contract["total_events"] == 207 and sum(contract["events_by_arm"].values()) == 207
    assert contract["events_by_arm"]["control"] == 9
    assert set(contract["code_sha256"]) == set(runner.MODULES)
    assert contract["outer_response_selection"] is False
    assert contract["hyperparameter_selection_scope"] == "inner_only_disjoint_reference_rows"


@pytest.mark.parametrize("field,bad", [("schema", "bad"), ("policy", {}), ("screen", {}), ("expression_decoded", True),
    ("new_raw_expression_allowed", True), ("automatic_promotion", 0), ("submission_allowed", True), ("cpu_only", False),
    ("stage", "pretrain"), ("wandb_mode", "online"), ("total_events", 206), ("events_by_arm", {}),
    ("outer_response_selection", True), ("hyperparameter_selection_scope", "outer")])
def test_registration_contract_cannot_be_relaxed(registered, field, bad):
    path = registered["path"]; record = json.loads(path.read_text()); record[field] = bad
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): runner.validate_contract(path, runner.sha256_file(path))


@pytest.mark.parametrize("which", ("role", "parent", "protocol"))
def test_changed_upstream_or_protocol_bytes_rejected(registered, which):
    registered[which].write_text("Changed synthetic metadata bytes")
    with pytest.raises(ValueError): runner.validate_contract(registered["path"], registered["digest"])


@pytest.mark.parametrize("part", ("code", "missing_code", "nested", "outer", "inputs", "protocol"))
def test_rehashed_contract_binding_changes_fail_closed(registered, part):
    path = registered["path"]; record = json.loads(path.read_text())
    if part == "code": record["code_sha256"][runner.MODULES[0]] = "f"*64
    elif part == "missing_code": record["code_sha256"].pop(runner.MODULES[0])
    elif part == "nested": record["nested_registration"]["sha256"] = "f"*64
    elif part == "outer": record["registration"]["sha256"] = "f"*64
    elif part == "inputs": record["inputs"]["cache_sha256"] = "f"*64
    else: record["protocol"]["sha256"] = "f"*64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError): runner.validate_contract(path, runner.sha256_file(path))


@pytest.mark.parametrize("part", ("top", "nested", "outer", "inputs", "extra", "group_bool", "group_source", "group_target"))
def test_full_summary_provenance_and_original_group_roles_bound(registered, part):
    contract, roles, nested = runner.validate_contract(registered["path"], registered["digest"])
    value = summary("true", roles=roles, nested=nested)
    value["diagnostic_contract_sha256"] = registered["digest"]
    value["provenance"] = {"diagnostic_contract_sha256": registered["digest"],
        "role_manifest_sha256": contract["registration"]["sha256"],
        "nested_manifest_sha256": contract["nested_registration"]["sha256"], "inputs": contract["inputs"]}
    if part == "top": value["diagnostic_contract_sha256"] = "f"*64
    elif part == "nested": value["provenance"]["nested_manifest_sha256"] = "f"*64
    elif part == "outer": value["provenance"]["role_manifest_sha256"] = "f"*64
    elif part == "inputs": value["provenance"]["inputs"] = {}
    elif part == "extra": value["provenance"]["unauthorized"] = True
    else:
        row = value["folds"][0]["groups"][0]
        if part == "group_bool": row["group"] = True
        elif part == "group_source": row["source"] = "wrong"
        else: row["target"] = "wrong"
    with pytest.raises(ValueError):
        runner.verify_summary(value, "true", registered["digest"], contract=contract, roles=roles, nested=nested)


def test_registration_is_never_overwritten(registered):
    with pytest.raises(FileExistsError):
        runner.register(registered["role"], registered["role_sha"], registered["protocol"], registered["path"].parent)


@pytest.mark.parametrize("host", ("login02.anvil", "aida.cac.cornell.edu", "cbsulogin.biohpc.cornell.edu", "cbsuvlaminck3-pretend"))
def test_fake_slurm_cannot_authorize_login_or_lookalike_host(tmp_path, monkeypatch, host):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: host)
    monkeypatch.setenv("SLURM_JOB_ID", "20539599")
    monkeypatch.setattr(runner, "load_cache", lambda *a, **k: pytest.fail("No decode on login"))
    with pytest.raises(RuntimeError): runner.run(tmp_path/"none", "a"*64, tmp_path/"run")
    assert not (tmp_path/"run").exists()


def audit_record(arm):
    return {**dict.fromkeys(runner.AUDIT_TRUE, True), "ridge_refits": 0,
            "inner_candidates_replayed": 0 if arm == "control" else 24,
            "out_of_fold_groups": 398, "out_of_fold_source_targets": 112}


@pytest.mark.parametrize("field", runner.AUDIT_TRUE)
def test_artifact_audit_true_flags_must_be_typed_bool(field):
    record = audit_record("true"); record[field] = 1
    with pytest.raises(ValueError): runner.verify_audit(record, "true")


@pytest.mark.parametrize("field", ("ridge_refits", "inner_candidates_replayed", "out_of_fold_groups", "out_of_fold_source_targets"))
def test_artifact_audit_counts_must_be_exact_integers(field):
    record = audit_record("true"); record[field] = float(record[field])
    with pytest.raises(ValueError): runner.verify_audit(record, "true")


def install_run(registered, monkeypatch, failure=None):
    monkeypatch.setattr(runner.socket, "gethostname", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setattr(runner.preparation, "load_feature_tables", lambda *a: {"inner": True})
    monkeypatch.setattr(runner.outer_preparation, "load_feature_tables", lambda *a: {"outer": True})
    arrays = {"synthetic": np.zeros(2)}
    row = {"excluded_targets": list(registered["roles"]["excluded_targets"])}
    if failure == "exclusions": row["excluded_targets"].pop()
    monkeypatch.setattr(runner, "load_cache", lambda *a: (arrays, row))
    def audit(output, value, *args):
        result = audit_record(value["arm"])
        if failure == "artifact_audit": result["passed"] = False
        return result
    monkeypatch.setattr(runner.model, "verify_artifacts", audit)
    sdks = []
    def tracker(*args, **kwargs):
        sdk = FakeSDK(log_error=failure == "log", finish_error=failure == "finish"); sdks.append(sdk)
        result = NestedValidationTracker(*args, sdk=sdk, **kwargs)
        if failure == "duplicate_run_ids": result.record["run_id"] = "same_id"
        return result
    monkeypatch.setattr(runner, "NestedValidationTracker", tracker)
    def fit(arrays, roles, outer_tables, nested, inner_tables, arm, output, provenance, progress=None):
        assert not arrays["synthetic"].flags.writeable
        assert roles == registered["roles"] and nested == registered["nested"]
        assert outer_tables == {"outer": True} and inner_tables == {"inner": True}
        if failure == "model": raise RuntimeError("Synthetic model failure")
        value = summary(arm, .98 if arm == "true" else 1., .09 if arm == "true" else .1, roles=roles, nested=nested)
        value.update(diagnostic_contract_sha256=provenance["diagnostic_contract_sha256"], provenance=provenance)
        for number, report in enumerate(value["folds"]):
            for inner in report["inner_selection"]["candidates"]:
                progress({"phase": "inner", "held_outer_source": report["held_source"], "regularization": inner["regularization"],
                    "metrics": inner["metrics"], "training_diagnostics": {}, "fold_index": number})
            if not (failure == "events" and number == 7):
                progress({"phase": "outer", "held_outer_source": report["held_source"], "regularization": report["regularization"],
                    "metrics": report["metrics"], "training_diagnostics": {}, "fold_index": number})
        if failure == "provenance": value["provenance"] = {"diagnostic_contract_sha256": "f"*64}
        if failure == "final_protocol" and arm == runner.model.ARMS[-1]: registered["protocol"].write_text("Changed during synthetic run")
        output.mkdir(); runner.dump_new(output/"summary.json", value)
        return value
    monkeypatch.setattr(runner.model, "run_arm", fit)
    return sdks


def test_mock_seven_arm_run_has_exact_207_events_and_consistent_completion(registered, tmp_path, monkeypatch):
    sdks = install_run(registered, monkeypatch); output = tmp_path/"run"
    assert runner.run(registered["path"], registered["digest"], output) == 0
    suite = json.loads((output/"suite.json").read_text()); complete = json.loads((output/"complete.json").read_text())
    assert suite["completed_at"] == complete["completed_at"]
    assert complete["suite_sha256"] == runner.sha256_file(output/"suite.json")
    assert complete["screen_passed"] is True
    assert suite["tracking_events"] == complete["tracking_events"] == 207
    assert suite["tracking_errors"] == complete["tracking_errors"] == 0
    for key in ("promoted", "submission_performed", "cloud_synced"):
        assert suite[key] is complete[key] is False
    assert all(suite[k] is False for k in ("pretraining_performed", "posttraining_performed", "new_raw_expression_read", "gpu_used", "download_performed"))
    assert len(sdks) == len(suite["runs"]) == 7
    for arm, sdk in zip(runner.model.ARMS, sdks):
        count = 9 if arm == "control" else 33
        assert len(sdk.run.logged) == count and sdk.initialized[0]["mode"] == "offline"
        assert [data["analysis/event_index"] for data, _ in sdk.run.logged] == list(range(1, count+1))
    with pytest.raises(FileExistsError): runner.run(registered["path"], registered["digest"], output)


@pytest.mark.parametrize("failure", ("model", "log", "finish", "events", "provenance", "artifact_audit", "exclusions",
                                     "duplicate_run_ids", "final_protocol"))
def test_failed_or_partial_run_never_leaves_completion(registered, tmp_path, monkeypatch, failure):
    sdks = install_run(registered, monkeypatch, failure); output = tmp_path/"run"
    with pytest.raises((ValueError, RuntimeError)): runner.run(registered["path"], registered["digest"], output)
    assert not (output/"complete.json").exists() and not (output/"suite.json").exists()
    if failure in {"model", "provenance", "artifact_audit"}:
        receipt = json.loads((output/"tracking/control/tracking.json").read_text())
        assert receipt["execution_status"] == "failed" and receipt["training_exit_code"] == 1
    if failure == "exclusions": assert sdks == []
