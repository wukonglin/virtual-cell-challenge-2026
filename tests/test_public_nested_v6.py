"""Two-gene synthetic nested selection and saved-artifact replay regression."""
import copy
import json
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_nested_v6 as v
from public_flow_conditioning import FeatureTable
from test_public_sparse_hurdle_v4 import fixture as sparse_fixture


def fixture():
    arrays, registration, _ = sparse_fixture()
    controls, control_groups, control_rows = [], [], []
    contexts, context_groups, context_rows, context_features = [], [], [], []
    treated_rows, pools = [], {}
    for group, (source, target, batch) in enumerate(zip(arrays["sources"], arrays["targets"], arrays["batches"])):
        sid, bid = (0 if source == "J" else 1000000), int(batch[-1])
        offset = sid + 10000 * bid
        reference = np.asarray([[0. if i % 2 == 0 else .1 + .02 * i,
            0. if i % 3 == 0 else .3 + .01 * i] for i in range(32)], dtype=np.float32)
        reference[reference > 0] += .02 * bid + (.1 if source == "K" else 0.)
        contextual = reference.copy()
        contextual[contextual > 0] += .03
        controls.append(reference); control_groups.extend([group] * 32)
        contexts.append(contextual); context_groups.extend([group] * 32)
        rows, xrows = np.arange(offset, offset+32), np.arange(offset+100, offset+132)
        control_rows.extend(rows); context_rows.extend(xrows)
        context_features.append(np.r_[contextual.mean(0), contextual.std(0)])
        ti = np.flatnonzero(arrays["group"] == group)
        treated_rows.extend(sid + 100000 + (ord(target)-ord("A"))*1000 + bid*100 + np.arange(len(ti)))
        pools[(str(source), str(batch))] = {"source": str(source), "batch": str(batch),
            "fit_reference_rows": rows[:16].tolist(), "tuning_anchor_rows": rows[16:].tolist(),
            "context_rows": xrows.tolist(), "sham_rows": list(range(offset+200, offset+208))}
    arrays.update(control=np.concatenate(controls), control_group=np.asarray(control_groups, dtype=np.int64),
        control_rows=np.asarray(control_rows, dtype=np.int64), context_control=np.concatenate(contexts),
        context_control_group=np.asarray(context_groups, dtype=np.int64), context_control_rows=np.asarray(context_rows, dtype=np.int64),
        context=np.asarray(context_features), treated_rows=np.asarray(treated_rows, dtype=np.int64))
    all_targets = ("A", "B", "C", "D")
    values = np.array([[1., .2], [.3, 1.], [.5, .8], [.9, .7]])
    table = FeatureTable("go", all_targets, ("term1", "term2"), values, "synthetic-outer", "a" * 64)
    outer_tables, inner_tables, inner_folds = {}, {}, {}
    for outer in registration["folds"]:
        name, fit = outer["fold_id"], outer["fit_targets"]
        outer_tables[name] = table
        inner_tables[name] = FeatureTable("go", tuple(fit), ("term1", "term2"),
            values[[all_targets.index(t) for t in fit]], "synthetic-inner-"+name, "b" * 64)
        inner_folds[name] = {"fit_targets": fit[:1], "tuning_targets": fit[1:], "directions": []}
    nested = {"inner_folds": inner_folds, "anchor_pools": [pools[key] for key in sorted(pools)]}
    return arrays, registration, outer_tables, nested, inner_tables


def first(arrays, registration, nested):
    outer = v.v3.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]
    inner = v.inner_fold(arrays, outer, nested)
    return outer, inner, v.inner_view(arrays, inner, nested)


def execute(path, arm="true"):
    args = fixture()
    summary = v.run_arm(*args, arm, path, {"diagnostic_contract_sha256": "c" * 64})
    return args, summary


@pytest.mark.parametrize("arm", v.ARMS)
def test_all_arms_complete_and_replay_without_refits(tmp_path, monkeypatch, arm):
    args, summary = execute(tmp_path / arm, arm)
    assert summary["hyperparameter_selection"] is (arm != "control")
    assert summary["inner_selection_performed"] is (arm != "control")
    assert summary["outer_response_selection"] is False
    events = [json.loads(line) for line in (tmp_path / arm / "steps.jsonl").read_text().splitlines()]
    assert len(events) == (4 if arm == "control" else 16)
    for report in summary["folds"]:
        candidates = report["inner_selection"]["candidates"]
        assert len(candidates) == (0 if arm == "control" else 3)
        assert report["regularization"] == report["inner_selection"]["selected_regularization"]
        if candidates:
            assert report["regularization"] == v.select_regularization(candidates)
    monkeypatch.setattr(v.kernel, "fit_kernel_decoder", lambda *a, **k: pytest.fail("Replay refit the model"))
    monkeypatch.setattr(np.linalg, "solve", lambda *a, **k: pytest.fail("Replay solved the model"))
    audit = v.verify_artifacts(tmp_path / arm, summary, *args)
    assert audit["passed"] and audit["ridge_refits"] == 0
    assert audit["inner_selection_replayed"] and audit["disjoint_inner_references_verified"]
    assert audit["inner_candidates_replayed"] == (0 if arm == "control" else 12)
    # The same audit must also work on the JSON representation actually saved.
    saved = json.loads((tmp_path / arm / "summary.json").read_text())
    assert v.verify_artifacts(tmp_path / arm, saved, *args)["passed"]


def test_lambda_exact_tie_prefers_strongest_regularization():
    candidates = [{"regularization": x, "metrics": {"target_pooled_mse": .5}} for x in v.GRID]
    assert v.select_regularization(candidates) == 10.0
    candidates[0]["metrics"]["target_pooled_mse"] = .499999999999
    assert v.select_regularization(candidates) == .1


@pytest.mark.parametrize("bad", [True, -1., float("nan"), float("inf"), "0.5"])
def test_nonfinite_or_untyped_inner_scores_rejected(bad):
    candidates = [{"regularization": x, "metrics": {"target_pooled_mse": .5}} for x in v.GRID]
    candidates[0]["metrics"]["target_pooled_mse"] = bad
    with pytest.raises(ValueError): v.select_regularization(candidates)


@pytest.mark.parametrize("part", ["outer_response", "outer_reference", "outer_context", "tuning_treated", "tuning_anchor", "context_donor"])
def test_forbidden_data_does_not_change_inner_fitted_state(part):
    arrays, registration, _, nested, inner_tables = fixture()
    outer, inner, view = first(arrays, registration, nested)
    _, _, before = v.one_fit(view, inner, inner_tables[outer.target_fold], "true", .1, "before_")
    modified = {key: value.copy() for key, value in arrays.items()}
    outer_forbidden = set(outer.held_target_indices_all_sources) | {i for i, s in enumerate(arrays["sources"]) if s == outer.held_source}
    if part == "outer_response": modified["treated"][np.isin(arrays["group"], list(outer_forbidden))] += 200
    elif part == "outer_reference": modified["control"][np.isin(arrays["control_group"], list(outer_forbidden))] += 100
    elif part == "outer_context":
        modified["context"][list(outer_forbidden)] += 99
        modified["context_control"][np.isin(arrays["context_control_group"], list(outer_forbidden))] += 77
    elif part == "tuning_treated": modified["treated"][np.isin(arrays["group"], inner.evaluation_indices)] += 300
    elif part == "tuning_anchor":
        ids = {r for p in nested["anchor_pools"] if p["source"] == inner.fit_source for r in p["tuning_anchor_rows"]}
        modified["control"][np.isin(arrays["control_rows"], list(ids))] += 400
    else: modified["context_control"] += 500
    after_view = v.inner_view(modified, inner, nested)
    _, _, after = v.one_fit(after_view, inner, inner_tables[outer.target_fold], "true", .1, "after_")
    for key in before:
        np.testing.assert_array_equal(before[key], after[key], err_msg=key)


def test_inner_reference_identities_disjoint_and_prior_excludes_context():
    arrays, registration, _, nested, _ = fixture()
    _, fold, view = first(arrays, registration, nested)
    fit = np.isin(view["control_group"], fold.fit_indices)
    tune = np.isin(view["control_group"], fold.evaluation_indices)
    assert not set(view["control_rows"][fit]) & set(view["control_rows"][tune])
    assert len(view["control_rows"][fit]) == len(fold.fit_indices) * 16
    assert len(view["control_rows"][tune]) == len(fold.evaluation_indices) * 16
    prior = v.positive_prior(view, fold)
    assert not set(prior.original_rows) & set(view["context_control_rows"])
    assert not set(prior.original_rows) & set(view["control_rows"][tune])
    assert set(prior.pool_kind) == {1, 2}
    assert set(prior.source_ids) == {fold.fit_source}


@pytest.mark.parametrize("part", ["duplicate_pool", "overlap", "wrong_original", "bool_original", "group_rows_differ"])
def test_invalid_global_reference_registration_rejected(part):
    arrays, registration, _, nested, _ = fixture()
    outer = v.v3.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]
    fold = v.inner_fold(arrays, outer, nested)
    pool = next(p for p in nested["anchor_pools"] if p["source"] == fold.fit_source)
    if part == "duplicate_pool": nested["anchor_pools"].append(copy.deepcopy(pool))
    elif part == "overlap": pool["tuning_anchor_rows"][0] = pool["fit_reference_rows"][0]
    elif part == "wrong_original": pool["fit_reference_rows"][0] = 999999999
    elif part == "bool_original": pool["fit_reference_rows"][0] = True
    else:
        group = fold.fit_indices[0]
        ci = np.flatnonzero(arrays["control_group"] == group)
        arrays["control_rows"][ci[0]] += 900
    with pytest.raises(ValueError): v.inner_view(arrays, fold, nested)


@pytest.mark.parametrize("part", ["selected_lambda", "candidate_score", "candidate_order", "control_parent", "coefficient", "saved_lambda", "prior_original", "row_dtype"])
def test_rehashed_saved_state_or_selection_tampering_rejected(tmp_path, part):
    out = tmp_path / "true"
    args, summary = execute(out)
    assert v.verify_artifacts(out, summary, *args)["passed"]
    if part == "selected_lambda": summary["folds"][0]["inner_selection"]["selected_regularization"] = 999.
    elif part == "candidate_score": summary["folds"][0]["inner_selection"]["candidates"][0]["metrics"]["target_pooled_mse"] += .1
    elif part == "candidate_order": summary["folds"][0]["inner_selection"]["candidates"].reverse()
    else:
        filename = "predictions" if part in {"control_parent", "row_dtype"} else "weights"
        with np.load(out / (filename + ".npz"), allow_pickle=False) as z:
            payload = {k: z[k].copy() for k in z.files}
        if part == "control_parent": payload["fold_0_inner_0_control_parent_cache_rows"][0] += 1
        elif part == "coefficient": payload["fold_0_inner_0_kernel_dual_coefficients"].flat[0] += .1
        elif part == "saved_lambda": payload["fold_0_inner_0_kernel_regularization"] = np.asarray(10.)
        elif part == "prior_original": payload["fold_0_inner_0_prior_original_rows"][0] += 1
        elif part == "row_dtype": payload["fold_0_inner_0_control_parent_cache_rows"] = payload["fold_0_inner_0_control_parent_cache_rows"].astype(float)
        # Synthetic mutation must rehash bytes, exercising semantic replay.
        temporary = out / ("tampered-" + filename + ".npz")
        digest = v.save_arrays(temporary, payload)
        temporary.replace(out / (filename + ".npz"))
        summary[filename + "_sha256"] = digest
    with pytest.raises(ValueError): v.verify_artifacts(out, summary, *args)


def test_context_donor_remains_available_without_prior_fitting():
    # Direct frozen output demonstrates this is an inference donor, not fitted data.
    pred = v.sparse.sparse_anchor(np.zeros((16, 2)), np.full((32, 2), .7), np.zeros(2),
        np.full(2, 4.), np.zeros(2), control_rows=np.arange(16), context_rows=np.arange(100,132),
        group_id=0, genes=("g1", "g2"))
    assert np.any(pred.nonnegative > 0)
    assert set(pred.donor_original_rows[pred.donor_original_rows >= 0]) <= set(range(100,132))
