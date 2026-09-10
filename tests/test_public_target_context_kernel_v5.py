"""Synthetic-only dual/primal, role leakage and safe restoration checks."""
from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import public_target_context_kernel_v5 as v
import public_joint_target_source_v3 as v3
import public_sparse_hurdle_v4 as v4
from public_flow_conditioning import FeatureTable, align_target_features
from test_public_sparse_hurdle_v4 import fixture


def inputs(arm="true"):
    arrays, registration, tables = fixture()
    fold = v3.joint_folds(tuple(arrays["targets"]), tuple(arrays["sources"]), registration)[0]
    aligned, receipt = v3.role_features(tables[fold.target_fold], fold.fit_targets, fold.held_targets,
        arm=v.feature_arm(arm), fold_id=fold.target_fold)
    prior = v4.training_positive_prior(arrays, fold)
    response = v4.training_responses(arrays, fold, prior)
    kwargs = {"targets": tuple(str(arrays["targets"][i]) for i in fold.fit_indices),
        "sources": tuple(str(arrays["sources"][i]) for i in fold.fit_indices),
        "context": arrays["context"][list(fold.fit_indices)], "response": response,
        "aligned": aligned, "arm": arm, "fold": fold, "representation": v.REPRESENTATION}
    return kwargs, arrays, tables, prior, receipt


def explicit_features(c, z, arm):
    if arm == "control": return np.empty((len(c), 0))
    pieces = [np.ones((len(c), 1)), c, np.einsum("ni,nj->nij", c, c).reshape(len(c), -1)]
    if arm not in {"control", "context"}: pieces.append(z)
    if arm not in {"control", "context", "additive"}:
        pieces.append(np.einsum("ni,nj->nij", z, c).reshape(len(c), -1))
    return np.column_stack(pieces)


def restore(model, kwargs, saved=None):
    return v.restore_kernel_decoder(v.decoder_arrays(model) if saved is None else saved, arm=kwargs["arm"],
        fold=kwargs["fold"], aligned=kwargs["aligned"], training_targets=kwargs["targets"],
        training_sources=kwargs["sources"], context=kwargs["context"], response=kwargs["response"])


def test_policy_exact_json_roundtrip_and_seven_arms():
    assert v.policy() == json.loads(json.dumps(v.policy()))
    assert len(v.ARMS) == 7 and v.RIDGE_LAMBDA == 10
    assert v.feature_arm("additive") == "true"
    assert v.policy()["capacity_matching_claimed"] is False
    assert v.policy()["hyperparameter_selection"] is False
    assert v.policy()["count_emitter"] is False


@pytest.mark.parametrize("arm", v.ARMS)
def test_kernel_equals_explicit_ordered_tensor_feature_gram(arm):
    rng = np.random.default_rng(20260925)
    left, right = rng.normal(size=(5, 3)), rng.normal(size=(7, 3))
    zl, zr = (rng.normal(size=(5, 2)), rng.normal(size=(7, 2))) if arm not in {"control", "context"} else (np.empty((5, 0)), np.empty((7, 0)))
    kernel = v.kernel_matrix(left, right, zl, zr, arm=arm)
    expected = explicit_features(left, zl, arm) @ explicit_features(right, zr, arm).T
    np.testing.assert_allclose(kernel, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("arm", v.ARMS)
def test_dual_predictions_equal_explicit_weighted_primal_ridge(arm):
    kw, arrays, _, _, _ = inputs(arm)
    model = v.fit_kernel_decoder(**kw)
    held = kw["fold"].evaluation_indices[0]
    aligned = v3._subset(kw["aligned"], (str(arrays["targets"][held]),))
    context = arrays["context"][held:held+1]
    prediction = model.predict_delta(aligned, context, is_ntc=np.array([False]))
    c = (context-model.context_mean)/model.context_scale/np.sqrt(context.shape[1])
    z = v._target_features(model.target_transform, aligned) if model.target_transform else np.empty((1, 0))
    train_phi = explicit_features(model.training_context_features, model.training_target_features, arm)
    test_phi = explicit_features(c, z, arm)
    if arm == "control": expected = np.zeros_like(prediction)
    else:
        w = model.training_weights
        beta = np.linalg.solve(train_phi.T @ (w[:, None]*train_phi) + 10*np.eye(train_phi.shape[1]),
                               train_phi.T @ (w[:, None]*kw["response"]))
        expected = test_phi @ beta
    np.testing.assert_allclose(prediction, expected, rtol=1e-11, atol=1e-13)


def test_negative_cross_kernel_entries_are_not_clipped():
    result = v.kernel_matrix(np.zeros((1, 1)), np.zeros((1, 1)), np.array([[2.]]), np.array([[-2.]]), arm="additive")
    assert result[0, 0] == -3


def test_penalized_constant_uses_same_lambda_and_normalized_weights():
    kw, _, _, _, _ = inputs("context")
    kw["context"] = np.ones_like(kw["context"])
    kw["response"] = np.array([[1., 2.], [2., 4.], [3., 6.], [4., 8.]])
    model = v.fit_kernel_decoder(**kw)
    aligned = v3._subset(kw["aligned"], ("A",))
    prediction = model.predict_delta(aligned, kw["context"][:1], is_ntc=np.array([False]))
    expected = (model.training_weights @ kw["response"])/11
    np.testing.assert_allclose(prediction[0], expected)
    metrics = v.training_diagnostics(model, kw["response"])
    assert metrics["kernel_trace"] == pytest.approx(1)
    assert metrics["effective_df"] == pytest.approx(1/11)


@pytest.mark.parametrize("arm", v.ARMS)
def test_ntc_exact_zero_and_v4_sparse_evaluation_compatibility(arm):
    kw, arrays, _, prior, _ = inputs(arm)
    model = v.fit_kernel_decoder(**kw)
    features = v3._subset(kw["aligned"], ("A", "B"))
    predicted = model.predict_delta(features, arrays["context"][:2], is_ntc=np.array([False, True]))
    np.testing.assert_array_equal(predicted[1], np.zeros(predicted.shape[1]))
    if arm == "control": np.testing.assert_array_equal(predicted, np.zeros_like(predicted))
    report, output = v4.evaluate_fold(arrays, kw["fold"], kw["aligned"], model, prior)
    assert report["exact_ntc_identity_verified"] and report["exact_zero_effect_identity_verified"]
    assert np.isfinite(output["nonnegative"]).all() and np.all(output["nonnegative"] >= 0)


@pytest.mark.parametrize("arm", ("additive", "true", *v.ARMS[4:]))
def test_held_annotations_do_not_fit_moments_weights_or_dual_coefficients(arm):
    kw, _, tables, _, _ = inputs(arm)
    before = v.fit_kernel_decoder(**kw)
    # Fold0 fits C/D; A/B are held and cannot affect training moments.
    changed = FeatureTable("go", ("A", "C"), ("term1", "term2"), np.array([[999., 321.], [.5, .8]]), "synthetic", "b"*64)
    kw["aligned"], _ = v3.role_features(changed, kw["fold"].fit_targets, kw["fold"].held_targets,
        arm=v.feature_arm(arm), fold_id=kw["fold"].target_fold)
    after = v.fit_kernel_decoder(**kw)
    for key in ("context_mean", "context_scale", "training_context_features", "training_target_features", "training_weights", "dual_coefficients"):
        np.testing.assert_array_equal(getattr(before, key), getattr(after, key))
    np.testing.assert_array_equal(before.target_transform.mean, after.target_transform.mean)


@pytest.mark.parametrize("mutation", ["target", "source", "overlap", "index_overlap"])
def test_admission_precedes_any_expression_or_response_read(mutation):
    class DoNotRead:
        def __array__(self, *args, **kwargs): raise AssertionError("Arrays read before admission")
    kw, _, _, _, _ = inputs()
    kw["context"] = kw["response"] = DoNotRead()
    if mutation == "target": kw["targets"] = ("A", "A", "D", "D")
    elif mutation == "source": kw["sources"] = (kw["fold"].held_source,)*4
    elif mutation == "overlap": kw["fold"] = replace(kw["fold"], held_targets=("C", "A"))
    else: kw["fold"] = replace(kw["fold"], held_target_indices_all_sources=kw["fold"].fit_indices)
    with pytest.raises(ValueError, match="Unauthorized"):
        v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("arm", v.ARMS[4:])
def test_shuffle_preserves_value_mask_packets_within_roles(arm):
    kw, _, tables, _, receipt = inputs(arm)
    original, _ = v3.role_features(tables[kw["fold"].target_fold], kw["fold"].fit_targets, kw["fold"].held_targets,
        arm="true", fold_id=kw["fold"].target_fold)
    for ids in (kw["fold"].fit_targets, kw["fold"].held_targets):
        assert {receipt["donors"][t] for t in ids} == set(ids)
        def packets(aligned):
            selected = v3._subset(aligned, ids)
            return sorted(tuple(row) for row in np.column_stack((selected.values, selected.present, selected.unknown_target)))
        assert packets(original) == packets(kw["aligned"])


def test_interaction_recovers_toy_sign_not_available_to_additive_kernel():
    fold = v3.JointFold("toy", "target_fold_0", "K", "J", ("A", "B"), ("C", "D"), (0, 1, 2, 3), (4, 5), (4, 5), ())
    table = FeatureTable("go", ("A", "B", "C", "D"), ("term",), np.array([[-1.], [1.], [-1.], [1.]]), "synthetic", "c"*64)
    aligned = align_target_features(("A", "B", "C", "D"), [table])
    context = np.array([[-1.], [1.], [-1.], [1.]])
    response = np.array([[1., 0.], [-1., 0.], [-1., 0.], [1., 0.]])
    predictions = {}
    for arm in ("additive", "true"):
        model = v.fit_kernel_decoder(("A", "A", "B", "B"), ("K",)*4, context, response,
            aligned=aligned, arm=arm, fold=fold)
        predictions[arm] = model.predict_delta(v3._subset(aligned, ("C", "D")), np.array([[1.], [1.]]), is_ntc=np.array([False, False]))[:, 0]
    np.testing.assert_allclose(predictions["additive"], [0., 0.], atol=1e-15)
    assert predictions["true"][0] < 0 < predictions["true"][1]


@pytest.mark.parametrize("change", ["nan_context", "inf_response", "complex", "odd_heads", "missing_group", "wrong_repr", "arm"])
def test_invalid_fit_arrays_refused(change):
    kw, _, _, _, _ = inputs()
    if change == "nan_context": kw["context"][0, 0] = np.nan
    elif change == "inf_response": kw["response"][0, 0] = np.inf
    elif change == "complex": kw["response"] = kw["response"].astype(complex)
    elif change == "odd_heads": kw["response"] = kw["response"][:, :3]
    elif change == "missing_group": kw["context"] = kw["context"][:-1]
    elif change == "wrong_repr": kw["representation"] = "raw_counts"
    else: kw["arm"] = "tuned"
    with pytest.raises(ValueError): v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("bad", [np.array([1., 2.]), np.empty((2, 0)), np.full((2, 2), np.nan), np.full((2, 2), 1e200)])
def test_bad_kernel_context_shapes_or_overflow_refused(bad):
    with pytest.raises(ValueError): v.kernel_matrix(bad, bad, arm="context")


def test_kernel_target_axes_and_ignored_nonempty_features_refused():
    with pytest.raises(ValueError, match="axes"):
        v.kernel_matrix(np.ones((2, 2)), np.ones((3, 2)), np.ones((1, 2)), np.ones((3, 2)), arm="true")
    with pytest.raises(ValueError, match="No target"):
        v.kernel_matrix(np.ones((2, 2)), np.ones((3, 2)), np.ones((2, 2)), np.ones((3, 2)), arm="context")


@pytest.mark.parametrize("arm", v.ARMS)
def test_saved_reconstruction_is_safe_and_never_refits(monkeypatch, arm):
    kw, arrays, _, _, _ = inputs(arm); model = v.fit_kernel_decoder(**kw)
    saved = v.decoder_arrays(model)
    assert set(saved) == set(v.decoder_keys(arm))
    assert all(not np.asarray(x).dtype.hasobject for x in saved.values())
    def forbidden(*args, **kwargs): raise AssertionError("Restoration must not solve or refit")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(v, "fit_kernel_decoder", forbidden)
    restored = restore(model, kw, saved)
    features = v3._subset(kw["aligned"], ("A", "B"))
    ctx = arrays["context"][:2]
    np.testing.assert_array_equal(model.predict_delta(features, ctx, is_ntc=np.zeros(2, dtype=bool)),
                                  restored.predict_delta(features, ctx, is_ntc=np.zeros(2, dtype=bool)))


@pytest.mark.parametrize("key", ["dual_coefficients", "training_context_features", "training_target_features", "training_weights",
                                 "context_mean", "target_mean", "training_response_sha256"])
def test_mutated_saved_state_is_rejected(key):
    kw, _, _, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    if key == "training_response_sha256": saved[key] = np.asarray("f"*64)
    else: saved[key].flat[0] += .1
    with pytest.raises(ValueError, match="residual|mismatch"):
        restore(model, kw, saved)


def test_training_diagnostics_cannot_use_held_or_changed_response():
    kw, _, _, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    report = v.training_diagnostics(model, kw["response"])
    assert set(report) == {"effective_df", "kernel_trace", "occupancy_response_rms", "occupancy_fitted_rms", "amplitude_response_rms", "amplitude_fitted_rms"}
    assert all(np.isfinite(x) and x >= 0 for x in report.values())
    w = model.training_weights
    expected = np.sqrt(np.sum(w*np.mean(kw["response"][:, :2]**2, axis=1)))
    assert report["occupancy_response_rms"] == pytest.approx(expected)
    with pytest.raises(ValueError, match="differs"):
        v.training_diagnostics(model, kw["response"]+.1)


def test_model_arrays_readonly_and_prediction_does_not_mutate_normalization():
    kw, arrays, _, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    original = v.decoder_arrays(model)
    assert not model.dual_coefficients.flags.writeable
    assert not model.training_context_features.flags.writeable
    features = v3._subset(kw["aligned"], ("A",))
    model.predict_delta(features, np.full_like(arrays["context"][:1], 100.), is_ntc=np.array([False]))
    for key, value in original.items(): np.testing.assert_array_equal(v.decoder_arrays(model)[key], value)


def test_missing_go_new_target_is_not_ntc():
    kw, arrays, tables, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    unknown = align_target_features(("NEVER_SEEN",), [tables[kw["fold"].target_fold]])
    assert unknown.unknown_target[0]
    result = model.predict_delta(unknown, arrays["context"][:1], is_ntc=np.array([False]))
    assert np.isfinite(result).all()
    np.testing.assert_array_equal(model.predict_delta(unknown, arrays["context"][:1], is_ntc=np.array([True])), np.zeros_like(result))
