"""Synthetic-only joint-source/target regression tests; never read real cache."""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import public_joint_target_source_v3 as v
from public_flow_conditioning import FeatureTable


def fixture():
    targets, sources, batches, context, control, treated, cg, tg = [], [], [], [], [], [], [], []
    for source in ("J", "K"):
        for target_index, target in enumerate(("A", "B", "C", "D")):
            for batch in range(2):
                group = len(targets)
                targets.append(target); sources.append(source); batches.append(f"{source}{batch}")
                reference = np.array([[0., .4], [.2, .5], [.3, .6]], dtype=np.float32) + .05 * batch
                source_offset = .2 if source == "K" else 0
                reference[:, 1] += source_offset
                control.append(reference); cg.extend([group] * 3)
                treated.append(reference[:2] + np.array([.02 * target_index, .03 * batch], dtype=np.float32)); tg.extend([group] * 2)
                context.append(np.r_[reference.mean(0) + .01, reference.std(0)])
    arrays = {"targets": np.asarray(targets), "sources": np.asarray(sources), "batches": np.asarray(batches),
        "genes": np.array(["gene1", "gene2"]), "context": np.asarray(context), "control": np.concatenate(control),
        "treated": np.concatenate(treated), "control_group": np.asarray(cg), "group": np.asarray(tg)}
    registration = {"excluded_targets": ["DENIED"], "folds": []}
    for number, held in enumerate((("A", "B"), ("C", "D"))):
        fit = sorted(set(targets) - set(held)); directions = []
        for hs, fs in (("J", "K"), ("K", "J")):
            fi = [i for i, (t, s) in enumerate(zip(targets, sources)) if t in fit and s == fs]
            hi = [i for i, t in enumerate(targets) if t in held]
            ei = [i for i in hi if sources[i] == hs]
            oi = [i for i in range(len(targets)) if i not in set(fi) | set(hi)]
            directions.append({"direction_id": f"target_fold_{number}_{fs}_to_{hs}", "fit_source": fs, "held_source": hs,
                "fit_group_ids": fi, "joint_eval_group_ids": ei, "held_target_group_ids_all_sources": hi,
                "other_excluded_group_ids": oi})
        registration["folds"].append({"fold_id": f"target_fold_{number}", "fit_targets": fit, "held_targets": list(held), "directions": directions})
    table = FeatureTable("go", ("A", "B", "C"), ("term1", "term2"), np.array([[1., .2], [.3, 1.], [.5, .8]]), "synthetic", "a" * 64)
    return arrays, registration, {r["fold_id"]: table for r in registration["folds"]}


def fit_one(arm="true", arrays=None, registration=None, tables=None):
    a, r, ts = fixture()
    arrays, registration, tables = arrays or a, registration or r, tables or ts
    fold = v.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]
    aligned, receipt = v.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets, arm=arm, fold_id=fold.target_fold)
    return v.fit_cache_fold(arrays, fold, aligned, arm), fold, aligned, receipt


def test_fixed_policy_no_selection_or_counts():
    assert v.policy()["ridge_lambda"] == 10
    assert v.policy()["gain"] == 1
    assert not v.policy()["hyperparameter_selection"]
    assert not v.policy()["count_emitter"]
    assert len(v.ARMS) == 6


def test_joint_roles_exclude_held_targets_both_sources():
    a, r, _ = fixture()
    folds = v.joint_folds(tuple(a["targets"]), tuple(a["sources"]), r)
    assert len(folds) == 4
    for f in folds:
        assert set(f.fit_indices).isdisjoint(f.held_target_indices_all_sources)
        assert set(f.evaluation_indices) <= set(f.held_target_indices_all_sources)
        assert {a["sources"][i] for i in f.fit_indices} == {f.fit_source}
        assert {a["sources"][i] for i in f.evaluation_indices} == {f.held_source}
        assert set(f.fit_targets).isdisjoint(f.held_targets)


@pytest.mark.parametrize("key", ["fit_group_ids", "joint_eval_group_ids", "held_target_group_ids_all_sources", "other_excluded_group_ids"])
def test_forged_registered_group_lists_refused(key):
    a, r, _ = fixture(); r["folds"][0]["directions"][0][key] = []
    with pytest.raises(ValueError, match=key):
        v.joint_folds(tuple(a["targets"]), tuple(a["sources"]), r)


@pytest.mark.parametrize("change", ["overlap", "missing", "duplicate", "excluded"])
def test_bad_target_roles_refused(change):
    a, r, _ = fixture()
    if change == "overlap": r["folds"][0]["fit_targets"].append("A")
    if change == "missing": r["folds"] = r["folds"][:1]
    if change == "duplicate": r["folds"][1] = copy.deepcopy(r["folds"][0])
    if change == "excluded": r["excluded_targets"] = ["A"]
    with pytest.raises(ValueError): v.joint_folds(tuple(a["targets"]), tuple(a["sources"]), r)


@pytest.mark.parametrize("arm", v.ARMS)
def test_unseen_targets_predict_without_lookup_and_ntc_exact(arm):
    a, _, _ = fixture(); model, fold, aligned, _ = fit_one(arm)
    assert "A" not in model.training_targets
    features = v._subset(aligned, ("A", "B"))
    delta = model.predict_delta(features, a["context"][:2], is_ntc=np.array([False, True]))
    assert delta.shape == (2, 2) and np.isfinite(delta).all()
    np.testing.assert_array_equal(delta[1], [0, 0])
    if arm == "control": np.testing.assert_array_equal(delta, np.zeros((2, 2)))


@pytest.mark.parametrize("arm", v.ARMS)
def test_held_effects_and_held_reference_do_not_change_fit(arm):
    a, r, ts = fixture(); before, f, _, _ = fit_one(arm)
    forbidden = set(f.held_target_indices_all_sources) | {i for i, s in enumerate(a["sources"]) if s == f.held_source}
    a["treated"][np.isin(a["group"], list(forbidden))] += 100
    a["control"][np.isin(a["control_group"], [i for i, s in enumerate(a["sources"]) if s == f.held_source])] += 200
    a["context"][[i for i, s in enumerate(a["sources"]) if s == f.held_source]] += 50
    after, _, _, _ = fit_one(arm, a, r, ts)
    np.testing.assert_array_equal(before.coefficients, after.coefficients)
    np.testing.assert_array_equal(before.context_mean, after.context_mean)
    if before.target_transform:
        np.testing.assert_array_equal(before.target_transform.mean, after.target_transform.mean)


def test_training_reference_changes_response_but_context_is_not_response_reference():
    a, r, ts = fixture(); before, f, aligned, _ = fit_one()
    a["control"][np.isin(a["control_group"], f.fit_indices)] += .7
    after = v.fit_cache_fold(a, f, aligned, "true")
    assert not np.array_equal(before.coefficients, after.coefficients)
    # A context perturbation changes the design, not the response construction.
    a, r, ts = fixture(); a["context"] += 2
    response_seen = []
    original = v.fit_joint_decoder
    try:
        def capture(*args, **kwargs):
            response_seen.append(args[3].copy()); return original(*args, **kwargs)
        v.fit_joint_decoder = capture
        fit_one(arrays=a, registration=r, tables=ts)
        fit_one()
    finally: v.fit_joint_decoder = original
    np.testing.assert_array_equal(*response_seen)


@pytest.mark.parametrize("arm", ("true", *v.ARMS[3:]))
def test_held_annotations_and_masks_do_not_fit_moments_or_weights(arm):
    a, r, ts = fixture(); before, _, _, _ = fit_one(arm)
    changed = FeatureTable("go", ("A", "C"), ("term1", "term2"), np.array([[999., 321.], [.5, .8]]), "synthetic", "b" * 64)
    after, _, _, _ = fit_one(arm, a, r, {key: changed for key in ts})
    np.testing.assert_array_equal(before.context_mean, after.context_mean)
    np.testing.assert_array_equal(before.target_transform.mean, after.target_transform.mean)
    np.testing.assert_array_equal(before.target_transform.scale, after.target_transform.scale)
    np.testing.assert_array_equal(before.coefficients, after.coefficients)


@pytest.mark.parametrize("arm", v.ARMS[3:])
def test_shuffle_conserves_role_packets_and_coverage(arm):
    _, _, ts = fixture(); table = ts["target_fold_0"]
    original, _ = v.role_features(table, ("A", "B"), ("C", "D"), arm="true", fold_id="f")
    shuffled, receipt = v.role_features(table, ("A", "B"), ("C", "D"), arm=arm, fold_id="f")
    for ids in (("A", "B"), ("C", "D")):
        assert {receipt["donors"][t] for t in ids} == set(ids)
        def packets(value):
            x = v._subset(value, ids)
            return sorted(tuple(row) for row in np.column_stack((x.values, x.present, x.unknown_target)))
        assert packets(original) == packets(shuffled)
    assert receipt["roles_crossed"] is False
    for role in ("fit", "held"):
        counts = receipt["association_changes"][role]
        assert counts["donor_fixed_points"] + counts["donor_identity_changed"] == counts["targets"]
        assert counts["identical_consumed_packets"] + counts["consumed_packet_changed"] == counts["targets"]
        assert counts["changed_donor_but_identical_packet"] >= 0


def test_shuffle_mapping_order_and_direction_invariant():
    a, r, ts = fixture(); folds = v.joint_folds(tuple(a["targets"]), tuple(a["sources"]), r)
    receipts = []
    for f in folds[:2]:
        _, receipt = v.role_features(ts[f.target_fold], tuple(reversed(f.fit_targets)), f.held_targets,
                                    arm=v.ARMS[3], fold_id=f.target_fold)
        receipts.append(receipt)
    assert receipts[0]["donors"] == receipts[1]["donors"]
    _, reordered = v.role_features(ts[folds[0].target_fold], folds[0].fit_targets, tuple(reversed(folds[0].held_targets)),
                                  arm=v.ARMS[3], fold_id=folds[0].target_fold)
    assert receipts[0]["donors"] == reordered["donors"]


def test_held_target_admission_before_response_read():
    class DoNotRead:
        def __array__(self, *args, **kwargs): raise AssertionError("Response accessed before admission")
    a, _, _ = fixture(); _, f, aligned, _ = fit_one()
    with pytest.raises(ValueError, match="Unauthorized"):
        v.fit_joint_decoder(("A", "B"), (f.fit_source, f.fit_source), DoNotRead(), DoNotRead(),
            aligned=aligned, arm="true", fold=f, representation=v.REPRESENTATION)


def test_target_pool_vectors_before_square_not_batch_mse():
    rows = [{"source": "J", "target": "A", "batch": str(i), "control_mean": np.zeros(2),
        "treated_mean": np.zeros(2), "raw_mean": np.array([sign, 0.]), "predicted_mean": np.array([sign, 0.])}
        for i, sign in enumerate((1., -1.))]
    report = v.pooled_target_metrics(rows)[0]
    assert report["target_pooled_mse"] == 0
    assert np.mean([np.mean(r["predicted_mean"] ** 2) for r in rows]) == .5
    with pytest.raises(ValueError, match="Duplicate"):
        v.pooled_target_metrics(rows + rows[:1])


@pytest.mark.parametrize("arm", v.ARMS)
def test_synthetic_run_safe_outputs_group_maps_and_source_macro(tmp_path, arm):
    a, r, ts = fixture(); events = []
    result = v.run_arm(a, r, ts, arm, tmp_path / arm, {"diagnostic_contract_sha256": "a" * 64}, events.append)
    assert len(result["folds"]) == 4 and len(events) == 4
    assert len(result["source_metrics"]) == 2
    assert result["exact_ntc_identity_verified"] and result["exact_zero_effect_identity_verified"]
    assert not result["held_roles_used_for_fitting"] and not result["count_emitter"]
    assert result["aggregate"]["negative_fraction"] == 0
    expected = np.mean([s["metrics"]["target_pooled_mse"] for s in result["source_metrics"]])
    assert result["aggregate"]["target_pooled_mse"] == pytest.approx(expected)
    with np.load(tmp_path / arm / "predictions.npz", allow_pickle=False) as saved:
        for f in result["folds"]:
            p = f["array_prefix"]
            assert len(saved[p + "group"]) != len(saved[p + "treated_group"])
            np.testing.assert_array_equal(a["control_group"][saved[p + "control_cache_rows"]], saved[p + "group"])
            np.testing.assert_array_equal(a["group"][saved[p + "treated_cache_rows"]], saved[p + "treated_group"])
            assert np.all(saved[p + "nonnegative"] >= 0)
    with np.load(tmp_path / arm / "weights.npz", allow_pickle=False) as saved:
        assert all(not saved[k].dtype.hasobject for k in saved)
        if arm not in {"control", "context"}: assert "fold_0_target_mean" in saved
    assert json.loads((tmp_path / arm / "summary.json").read_text())["arm"] == arm
    with pytest.raises(FileExistsError):
        v.run_arm(a, r, ts, arm, tmp_path / arm, {"diagnostic_contract_sha256": "a" * 64})


def test_control_pooled_and_raw_exact_baseline(tmp_path):
    a, r, ts = fixture()
    result = v.run_arm(a, r, ts, "control", tmp_path / "control", {"diagnostic_contract_sha256": "a" * 64})
    assert result["aggregate"]["target_pooled_mse"] == result["aggregate"]["control_target_pooled_mse"]
    assert result["aggregate"]["raw_target_pooled_mse"] == result["aggregate"]["control_target_pooled_mse"]
    assert result["aggregate"]["projection_fraction"] == 0


def test_constant_fractional_features_stable():
    a, r, ts = fixture()
    table = FeatureTable("go", ("A", "B", "C", "D"), ("term",), np.full((4, 1), .1), "synthetic", "c" * 64)
    model, _, _, _ = fit_one("true", a, r, {k: table for k in ts})
    np.testing.assert_array_equal(model.target_transform.mean, [.1])
    np.testing.assert_array_equal(model.target_transform.scale, [1.])


def test_prediction_cannot_refit_context_or_target_transform():
    a, _, _ = fixture(); model, _, aligned, _ = fit_one()
    mean, scale, coefficients = model.context_mean.copy(), model.target_transform.scale.copy(), model.coefficients.copy()
    model.predict_delta(v._subset(aligned, ("A",)), np.full((1, 4), 100.), is_ntc=np.array([False]))
    np.testing.assert_array_equal(model.context_mean, mean)
    np.testing.assert_array_equal(model.target_transform.scale, scale)
    np.testing.assert_array_equal(model.coefficients, coefficients)


def test_saved_arrays_replay_unseen_target_predictions_without_pickle(tmp_path):
    a, r, ts = fixture()
    result = v.run_arm(a, r, ts, "true", tmp_path / "replay", {"diagnostic_contract_sha256": "a" * 64})
    with np.load(tmp_path / "replay" / "weights.npz", allow_pickle=False) as saved, np.load(tmp_path / "replay" / "predictions.npz", allow_pickle=False) as predicted:
        for fold in result["folds"]:
            p = fold["array_prefix"]
            target_ids = tuple(saved[p + "aligned_target_ids"].astype(str))
            values, present = saved[p + "aligned_values"], saved[p + "aligned_present"]
            feature = (values - saved[p + "target_mean"]) / saved[p + "target_scale"]
            feature[~present[:, 0]] = 0
            feature = np.column_stack((feature, present, ~present.any(1)))
            feature /= np.sqrt(feature.shape[1])
            held = predicted[p + "held_groups"]
            target_rows = [target_ids.index(str(a["targets"][i])) for i in held]
            context = (a["context"][held] - saved[p + "context_mean"]) / saved[p + "context_scale"]
            context /= np.sqrt(context.shape[1])
            design = np.column_stack((np.ones(len(held)), feature[target_rows], context))
            np.testing.assert_allclose(design @ saved[p + "coefficients"], predicted[p + "delta"], rtol=1e-14, atol=1e-14)


def test_missing_held_target_remains_in_evaluation():
    a, r, ts = fixture()
    fold = v.joint_folds(tuple(a["targets"]), tuple(a["sources"]), r)[2]
    aligned, _ = v.role_features(ts[fold.target_fold], fold.fit_targets, fold.held_targets, arm="true", fold_id=fold.target_fold)
    model = v.fit_cache_fold(a, fold, aligned, "true")
    missing = v._subset(aligned, ("D",))
    assert missing.unknown_target[0]
    delta = model.predict_delta(missing, a["context"][:1], is_ntc=np.array([False]))
    assert np.isfinite(delta).all()
    report, _ = v.evaluate_joint_fold(a, fold, aligned, model)
    assert "D" in {t["target"] for t in report["targets_report"]}
