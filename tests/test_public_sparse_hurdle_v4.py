"""Synthetic-only hurdle tests; no existing expression artifacts are opened."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_sparse_hurdle_v4 as v
import public_joint_target_source_v3 as v3
from test_public_joint_target_source_v3 import fixture as joint_fixture


def fixture():
    arrays, registration, tables = joint_fixture()
    control_rows, context_rows, treated_rows, context, context_groups = [], [], [], [], []
    for group, (source, target, batch) in enumerate(zip(arrays["sources"], arrays["targets"], arrays["batches"])):
        sid = 0 if source == "J" else 1000; bid = int(batch[-1]); tid = ord(target) - ord("A")
        ci = np.flatnonzero(arrays["control_group"] == group); ti = np.flatnonzero(arrays["group"] == group)
        control_rows.extend(sid + 10*bid + np.arange(len(ci)))
        context_rows.extend(sid + 100 + 10*bid + np.arange(len(ci)))
        treated_rows.extend(10000 + sid + 100*tid + 10*bid + np.arange(len(ti)))
        context.append(arrays["control"][ci] + .03); context_groups.extend([group]*len(ci))
    arrays.update(control_rows=np.asarray(control_rows, dtype=np.int64), treated_rows=np.asarray(treated_rows, dtype=np.int64),
        context_control_rows=np.asarray(context_rows, dtype=np.int64), context_control=np.concatenate(context),
        context_control_group=np.asarray(context_groups, dtype=np.int64))
    return arrays, registration, tables


def anchor(control, occ, amp, *, context=None, prior=None, rows=None, xrows=None, **kwargs):
    control = np.asarray(control, dtype=np.float64); width = control.shape[1]
    context = np.zeros_like(control) if context is None else np.asarray(context, dtype=np.float64)
    return v.sparse_anchor(control, context, np.ones(width) if prior is None else prior, occ, amp,
        control_rows=np.arange(len(control), dtype=np.int64) if rows is None else rows,
        context_rows=100+np.arange(len(context), dtype=np.int64) if xrows is None else xrows,
        group_id=7, genes=tuple(f"g{i}" for i in range(width)), **kwargs)


def first_fold(arrays, registration):
    return v3.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]


def test_policy_json_roundtrip_and_output_only_scope():
    assert v.policy() == json.loads(json.dumps(v.policy()))
    assert v.policy()["base_design"] == v3.policy()
    assert v.ARMS == v3.ARMS
    assert v.policy()["ridge_lambda"] == 10
    assert v.policy()["interactions"] is False and v.policy()["count_emitter"] is False


@pytest.mark.parametrize("ntc", [False, True])
def test_exact_zero_or_ntc_bypasses_all_smoothing(ntc):
    control = np.array([[0., .1], [.3, 0.], [.7, .8]])
    prediction = anchor(control, [99., -99.] if ntc else [0., 0.], [99., 99.] if ntc else [0., 0.], is_ntc=ntc)
    np.testing.assert_array_equal(prediction.nonnegative, control)
    np.testing.assert_array_equal(prediction.requested_occupancy, np.mean(control > 0, axis=0))
    assert prediction.metrics["exact_zero_effect_identity_verified"]
    assert prediction.metrics["unchanged_count_support_verified"]
    if ntc:
        assert prediction.metrics["exact_ntc_identity_verified"]
        assert prediction.metrics["occupancy_clip_fraction"] == 0


def test_per_gene_zero_head_identity():
    control = np.array([[0., .1], [.3, 0.], [.7, .8]])
    prediction = anchor(control, [0., -4.], [0., .7])
    np.testing.assert_array_equal(prediction.nonnegative[:, 0], control[:, 0])


@pytest.mark.parametrize("n", [2, 8, 32])
def test_same_empirical_occupancy_different_n_has_zero_response(n):
    reference = np.array([[0., 3., 0.], [2., 3., 0.]])
    treated = np.tile(reference, (n//2, 1))
    response = v.head_response(treated, reference, [2., 3., 0.])
    np.testing.assert_array_equal(response, np.zeros(6))


def test_response_formula_all_zero_all_positive_and_unsupported():
    treated = np.array([[0., 2., 0.], [0., 2., 0.]])
    reference = np.array([[1., 0., 0.]] * 8)
    response = v.head_response(treated, reference, [1., 2., 0.])
    expected_logit_gap = np.log(32.5/.5) - np.log(.5/32.5)
    assert response[0] == pytest.approx(-expected_logit_gap)
    assert response[1] == pytest.approx(expected_logit_gap)
    assert response[2] == 0 and response[5] == 0
    np.testing.assert_allclose(response[3:], [0., 0., 0.], atol=1e-15)


def test_magnitude_response_uses_fixed_prior_smoothing():
    treated = np.array([[0., 4.], [4., 8.]])
    reference = np.array([[0., 2.], [2., 4.], [0., 2.], [2., 4.]])
    prior = np.array([3., 5.])
    result = v.head_response(treated, reference, prior)
    mt = (treated.mean(0) + prior/32) / (np.mean(treated > 0, axis=0)+1/32)
    mc = (reference.mean(0) + prior/32) / (np.mean(reference > 0, axis=0)+1/32)
    np.testing.assert_allclose(result[2:], np.log(mt)-np.log(mc))


def test_prior_deduplicates_shared_original_anchor_rows():
    a, r, _ = fixture(); fold = first_fold(a, r)
    prior = v.training_positive_prior(a, fold)
    assert np.sum(prior.pool_kind == 1) == 6  # two batches, three controls, reused by targets
    assert np.sum(prior.pool_kind == 2) == 8  # two fit targets × two batches × two treated
    assert len(prior.original_rows) == 14
    assert len(set(zip(prior.source_ids, prior.original_rows))) == 14
    unique = {}
    for field, lab, row in (("control", "control_group", "control_rows"), ("treated", "group", "treated_rows")):
        for i in np.flatnonzero(np.isin(a[lab], fold.fit_indices)):
            unique[int(a[row][i])] = a[field][i]
    values = np.asarray(list(unique.values()))
    expected = np.array([values[:, g][values[:, g]>0].astype(np.float64).mean() for g in range(values.shape[1])])
    np.testing.assert_allclose(prior.values, expected)


@pytest.mark.parametrize("mutation", ["held_source", "held_target", "context"])
def test_prior_and_responses_exclude_forbidden_data(mutation):
    a, r, _ = fixture(); fold = first_fold(a, r)
    prior = v.training_positive_prior(a, fold); response = v.training_responses(a, fold, prior)
    if mutation == "held_source":
        groups = [i for i, s in enumerate(a["sources"]) if s == fold.held_source]
        a["control"][np.isin(a["control_group"], groups)] += 100
        a["treated"][np.isin(a["group"], groups)] += 200
    elif mutation == "held_target":
        a["treated"][np.isin(a["group"], fold.held_target_indices_all_sources)] += 300
    else:
        a["context_control"] += 400; a["context"] += 500
    after = v.training_positive_prior(a, fold)
    np.testing.assert_array_equal(prior.values, after.values)
    np.testing.assert_array_equal(response, v.training_responses(a, fold, after))


def test_prior_rejects_conflicting_duplicate_original_rows():
    a, r, _ = fixture(); fold = first_fold(a, r)
    ci = np.flatnonzero(np.isin(a["control_group"], fold.fit_indices))
    a["control"][ci[-1], 0] += .1
    with pytest.raises(ValueError, match="conflicting"):
        v.training_positive_prior(a, fold)


def test_prior_unsupported_gene_is_zero_not_fabricated():
    a, r, _ = fixture(); fold = first_fold(a, r)
    a["control"][:, 0] = 0; a["treated"][:, 0] = 0
    prior = v.training_positive_prior(a, fold)
    assert prior.values[0] == 0 and not prior.supported[0] and prior.positive_counts[0] == 0
    response = v.training_responses(a, fold, prior)
    np.testing.assert_array_equal(response[:, [0, 2]], np.zeros((len(fold.fit_indices), 2)))


def test_tiny_occupancy_shift_preserves_exact_original_values_and_support():
    control = np.r_[np.zeros(16), np.linspace(.1, 2., 16)][:, None]
    pred = anchor(control, [1e-7], [0.])
    assert pred.requested_occupancy[0] > .5
    assert pred.desired_positive_count[0] == 16
    np.testing.assert_array_equal(pred.nonnegative, control)
    assert pred.metrics["unchanged_count_fraction"] == 1


def test_amplitude_only_preserves_support_exactly():
    control = np.array([[0.], [.2], [0.], [.8]])
    pred = anchor(control, [0.], [np.log(2.)])
    np.testing.assert_array_equal(pred.nonnegative > 0, control > 0)
    np.testing.assert_array_equal(pred.requested_occupancy, [.5])
    np.testing.assert_allclose(pred.nonnegative, 2*control)


def test_fixed_ranks_make_nested_activation_and_donor_order_arm_independent():
    control = np.r_[.3, .8, np.zeros(6)][:, None]
    low = anchor(control, [1.5], [0.]); high = anchor(control, [4.], [0.])
    low_add = np.flatnonzero(low.donor_kind[:, 0] == 2)
    assert len(low_add) > 0 and low.desired_positive_count[0] < high.desired_positive_count[0]
    np.testing.assert_array_equal(low.donor_original_rows[low_add], high.donor_original_rows[low_add])
    np.testing.assert_array_equal(low.nonnegative[low_add], high.nonnegative[low_add])


def test_deactivation_only_keeps_original_positive_values_by_fixed_rank():
    control = np.arange(1., 9.)[:, None]
    pred = anchor(control, [-4.], [0.])
    keep = pred.nonnegative[:, 0] > 0
    np.testing.assert_array_equal(pred.nonnegative[keep], control[keep])
    assert pred.metrics["deactivation_count"] == np.sum(~keep)
    assert pred.metrics["activation_count"] == 0


@pytest.mark.parametrize("donor", ["context", "prior", "none"])
def test_zero_anchor_fallback_and_explicit_unresolved(donor):
    control = np.zeros((32, 1)); context = np.zeros((4, 1)); prior = [0.]
    if donor == "context": context[2, 0] = .7
    if donor == "prior": prior = [.9]
    pred = anchor(control, [4.], [0.], context=context, prior=prior)
    assert pred.desired_positive_count[0] > 0
    if donor == "none":
        np.testing.assert_array_equal(pred.nonnegative, control)
        assert pred.metrics["unresolved_activation_fraction"] == 1
        assert pred.metrics["positive_count_shortfall"] == pred.desired_positive_count[0]
        assert np.sum(pred.donor_kind == 5) > 0
    else:
        assert pred.realized_positive_count[0] == pred.desired_positive_count[0]
        assert pred.metrics["unresolved_activation_fraction"] == 0
        assert np.sum(pred.donor_kind == (3 if donor == "context" else 4)) > 0


def test_clamps_are_reported_not_silent():
    pred = anchor(np.ones((8, 2)), [9., -9.], [9., -9.])
    np.testing.assert_array_equal(pred.occupancy_delta, [4., -4.])
    np.testing.assert_allclose(pred.amplitude_delta, [np.log(4.), -np.log(4.)])
    assert pred.metrics["occupancy_clip_fraction"] == 1
    assert pred.metrics["amplitude_clip_fraction"] == 1
    assert np.max(pred.nonnegative) == 4
    assert np.isfinite(pred.nonnegative).all() and np.all(pred.nonnegative >= 0)


def test_ranking_and_values_are_invariant_to_cell_row_reordering():
    control = np.r_[.3, .8, np.zeros(6)][:, None]; order = np.array([7, 2, 3, 1, 5, 6, 0, 4])
    before = anchor(control, [4.], [.3])
    after = anchor(control[order], [4.], [.3], rows=np.arange(8)[order])
    reverse = np.argsort(order)
    np.testing.assert_array_equal(before.nonnegative, after.nonnegative[reverse])
    np.testing.assert_array_equal(before.donor_original_rows, after.donor_original_rows[reverse])


@pytest.mark.parametrize("change", ["negative", "nan", "rows", "overlap", "repr", "ntc"])
def test_bad_inputs_refused(change):
    control = np.ones((4, 1)); kwargs = {}; occ = [0.]
    if change == "negative": control[0] = -1
    if change == "nan": occ = [np.nan]
    if change == "rows": kwargs["rows"] = np.array([0, 0, 1, 2])
    if change == "overlap": kwargs["xrows"] = np.arange(4)
    if change == "repr": kwargs["representation"] = "raw_counts"
    if change == "ntc": kwargs["is_ntc"] = 1
    with pytest.raises(ValueError): anchor(control, occ, [0.], **kwargs)


def test_output_overflow_is_rejected():
    with pytest.raises(ValueError, match="overflow"):
        anchor(np.full((4, 1), 1e308), [0.], [2.])


def test_positive_underflow_is_rejected_not_silently_dropped():
    with pytest.raises(ValueError, match="underflow"):
        anchor(np.full((4, 1), np.nextafter(0., 1.)), [0.], [-2.])


def test_vector_first_occupancy_and_expression_pooling():
    rows = [{"source": "J", "target": "A", "batch": str(i), "control_mean": np.array([.5]),
        "treated_mean": np.array([.5]), "predicted_mean": np.array([.5+error]),
        "control_occupancy": np.array([.5]), "treated_occupancy": np.array([.5]), "predicted_occupancy": np.array([.5+error])}
        for i, error in enumerate((.2, -.2))]
    report = v.target_metrics(rows)[0]
    assert report["target_pooled_mse"] == 0
    assert report["target_pooled_occupancy_mse"] == 0


@pytest.mark.parametrize("arm", v.ARMS)
def test_synthetic_end_to_end_safe_artifacts_and_exact_replay(tmp_path, arm):
    a, r, tables = fixture(); events = []
    summary = v.run_arm(a, r, tables, arm, tmp_path/arm, {"diagnostic_contract_sha256": "a"*64}, events.append)
    assert len(events) == 4 and len(summary["folds"]) == 4
    assert summary["exact_ntc_identity_verified"] and summary["exact_zero_effect_identity_verified"]
    assert summary["aggregate"]["negative_fraction"] == 0
    assert not summary["count_emitter"]
    assert summary["aggregate"]["target_pooled_mse"] == pytest.approx(np.mean([s["metrics"]["target_pooled_mse"] for s in summary["source_metrics"]]))
    receipt = v.verify_artifacts(tmp_path/arm, summary, a, r, tables)
    assert receipt["passed"] and receipt["ridge_refits"] == 0
    saved_summary = json.loads((tmp_path/arm/"summary.json").read_text())
    assert v.verify_artifacts(tmp_path/arm, saved_summary, a, r, tables)["passed"]
    with np.load(tmp_path/arm/"predictions.npz", allow_pickle=False) as saved:
        assert len(saved["fold_0_group"]) != len(saved["fold_0_treated_group"])
        np.testing.assert_array_equal(a["control_rows"][saved["fold_0_control_cache_rows"]], saved["fold_0_control_original_rows"])
        np.testing.assert_array_equal(a["treated_rows"][saved["fold_0_treated_cache_rows"]], saved["fold_0_treated_original_rows"])
    if arm == "control":
        assert summary["aggregate"]["target_pooled_mse"] == summary["aggregate"]["control_target_pooled_mse"]
        assert summary["aggregate"]["zero_fraction"] == summary["aggregate"]["control_zero_fraction"]
    with pytest.raises(FileExistsError):
        v.run_arm(a, r, tables, arm, tmp_path/arm, {"diagnostic_contract_sha256": "a"*64})


@pytest.mark.parametrize("mutation", ["output", "coefficients", "donor", "prior", "heads", "rows", "metric", "pooled_metric"])
def test_rehashed_artifact_or_metric_mutations_rejected(tmp_path, mutation):
    a, r, tables = fixture(); out = tmp_path/mutation
    summary = v.run_arm(a, r, tables, "true", out, {"diagnostic_contract_sha256": "a"*64})
    if mutation in {"metric", "pooled_metric"}:
        if mutation == "metric": summary["folds"][0]["groups"][0]["model_mmd2"] += .01
        else: summary["source_metrics"][0]["metrics"]["target_pooled_occupancy_mse"] += .01
    else:
        filename = "weights.npz" if mutation in {"coefficients", "prior"} else "predictions.npz"
        with np.load(out/filename, allow_pickle=False) as saved: payload = {k: saved[k].copy() for k in saved}
        key = {"output": "nonnegative", "coefficients": "coefficients", "donor": "donor_original_rows",
            "prior": "positive_prior", "heads": "head_delta", "rows": "control_original_rows"}[mutation]
        payload["fold_0_"+key].flat[0] += 1
        # Synthetic temporary artifact replacement deliberately tests content-hash reauthorization.
        (out/filename).unlink()
        summary["weights_sha256" if filename == "weights.npz" else "predictions_sha256"] = v.save_arrays(out/filename, payload)
    with pytest.raises(ValueError, match="replay|equations|mismatch"):
        v.verify_artifacts(out, summary, a, r, tables)


def test_held_cells_never_change_two_head_coefficients():
    a, r, tables = fixture(); fold = first_fold(a, r)
    aligned, _ = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets, arm="true", fold_id=fold.target_fold)
    before, prior, response = v.fit_hurdle_fold(a, fold, aligned, "true")
    forbidden = set(fold.held_target_indices_all_sources) | {i for i, s in enumerate(a["sources"]) if s == fold.held_source}
    a["treated"][np.isin(a["group"], list(forbidden))] += 100
    a["control"][np.isin(a["control_group"], [i for i, s in enumerate(a["sources"]) if s == fold.held_source])] += 200
    after, after_prior, after_response = v.fit_hurdle_fold(a, fold, aligned, "true")
    np.testing.assert_array_equal(before.coefficients, after.coefficients)
    np.testing.assert_array_equal(prior.values, after_prior.values)
    np.testing.assert_array_equal(response, after_response)


def test_jointly_rehashed_coefficients_and_heads_cannot_bypass_fit_equations(tmp_path):
    a, r, tables = fixture(); out = tmp_path/"joint_mutation"
    summary = v.run_arm(a, r, tables, "true", out, {"diagnostic_contract_sha256": "a"*64})
    for filename, key, hashkey in (("weights.npz", "coefficients", "weights_sha256"),
                                   ("predictions.npz", "head_delta", "predictions_sha256")):
        with np.load(out/filename, allow_pickle=False) as saved: payload = {k: saved[k].copy() for k in saved}
        payload["fold_0_"+key].flat[0] += 1
        (out/filename).unlink()
        summary[hashkey] = v.save_arrays(out/filename, payload)
    with pytest.raises(ValueError, match="normal equations"):
        v.verify_artifacts(out, summary, a, r, tables)
