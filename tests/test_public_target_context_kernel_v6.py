"""Synthetic-only explicit-lambda, inner-role, overflow and replay tests."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
import public_target_context_kernel_v6 as v
import public_target_context_kernel_v5 as v5
import public_joint_target_source_v3 as v3
from test_public_target_context_kernel_v5 import inputs as original_inputs, explicit_features


def inputs(arm="true", regularization=1.0):
    kw, arrays, tables, prior, receipt = original_inputs(arm)
    return {**kw, "regularization": regularization}, arrays, tables, prior, receipt


def inner_inputs(arm="true", regularization=1.0):
    kw, arrays, tables, prior, receipt = inputs(arm, regularization)
    outer = kw["fold"]
    positions = [i for i, t in enumerate(kw["targets"]) if t == "C"]
    held = tuple(outer.fit_indices[i] for i, t in enumerate(kw["targets"]) if t == "D")
    kw["fold"] = v.InnerTargetFold("inner_0", "inner_target_0", outer.fit_source, outer.fit_source,
        ("C",), ("D",), tuple(outer.fit_indices[i] for i in positions), held, held, (), outer.fold_id)
    kw["targets"] = tuple(kw["targets"][i] for i in positions)
    kw["sources"] = tuple(kw["sources"][i] for i in positions)
    kw["context"] = kw["context"][positions]
    kw["response"] = kw["response"][positions]
    # Inner roles need their own packet assignment; slicing an outer shuffle
    # can move a held packet across the newly introduced calibration boundary.
    kw["aligned"], receipt = v3.role_features(tables[outer.target_fold], kw["fold"].fit_targets,
        kw["fold"].held_targets, arm=v.feature_arm(arm), fold_id=kw["fold"].target_fold)
    return kw, arrays, tables, prior, receipt


def restore(model, kw, *, saved=None, **changes):
    args = {"arm": kw["arm"], "fold": kw["fold"], "aligned": kw["aligned"],
        "training_targets": kw["targets"], "training_sources": kw["sources"],
        "context": kw["context"], "response": kw["response"], "regularization": kw["regularization"]}
    return v.restore_kernel_decoder(v.decoder_arrays(model) if saved is None else saved, **{**args, **changes})


def test_policy_is_json_stable_explicit_grid_and_no_selection():
    assert v.policy() == json.loads(json.dumps(v.policy()))
    assert v.REGULARIZATION_GRID == (0.1, 1.0, 10.0)
    assert v.policy()["hyperparameter_selection"] is False
    assert "ridge_lambda" not in v.policy()
    assert v.ARMS == v5.ARMS


@pytest.mark.parametrize("regularization", v.REGULARIZATION_GRID)
@pytest.mark.parametrize("arm", v.ARMS)
def test_grid_dual_equals_explicit_weighted_primal(regularization, arm):
    kw, arrays, _, _, _ = inputs(arm, regularization)
    model = v.fit_kernel_decoder(**kw)
    held = kw["fold"].evaluation_indices[0]
    annotations = v3._subset(kw["aligned"], (str(arrays["targets"][held]),))
    context = arrays["context"][held:held+1]
    prediction = model.predict_delta(annotations, context, is_ntc=np.array([False]))
    c = (context-model.context_mean)/model.context_scale/np.sqrt(context.shape[1])
    z = v._target_features(model.target_transform, annotations) if model.target_transform else np.empty((1, 0))
    train_phi = explicit_features(model.training_context_features, model.training_target_features, arm)
    test_phi = explicit_features(c, z, arm)
    if arm == "control": expected = np.zeros_like(prediction)
    else:
        w = model.training_weights
        beta = np.linalg.solve(train_phi.T@(w[:, None]*train_phi)+regularization*np.eye(train_phi.shape[1]),
                               train_phi.T@(w[:, None]*kw["response"]))
        expected = test_phi@beta
    np.testing.assert_allclose(prediction, expected, rtol=1e-10, atol=1e-13)
    assert model.regularization == regularization
    assert model.validation_kind == "outer_joint_target_source" and model.outer_fold_id == ""


@pytest.mark.parametrize("arm", v.ARMS)
def test_lambda10_preserves_v5_results_and_global_constant(arm):
    kw, arrays, _, _, _ = inputs(arm, 10.0)
    new = v.fit_kernel_decoder(**kw)
    old = v5.fit_kernel_decoder(**{k: x for k, x in kw.items() if k != "regularization"})
    np.testing.assert_array_equal(new.dual_coefficients, old.dual_coefficients)
    np.testing.assert_array_equal(new.training_context_features, old.training_context_features)
    assert dict(new.training_metrics) == dict(old.training_metrics)
    v.fit_kernel_decoder(**{**kw, "regularization": 0.1})
    assert v5.RIDGE_LAMBDA == 10.0


@pytest.mark.parametrize("regularization", [0, -1, .01, .5, 100, np.nan, np.inf, True, False, "1", None, np.float32(.1), np.array(1.0)])
def test_nonregistered_or_ambiguous_lambda_refused(regularization):
    kw, _, _, _, _ = inputs(regularization=regularization)
    with pytest.raises(ValueError, match="regularization|Regularization"):
        v.fit_kernel_decoder(**kw)


def test_lambda_has_no_implicit_default():
    kw, _, _, _, _ = inputs(); del kw["regularization"]
    with pytest.raises(TypeError, match="regularization"):
        v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("regularization", v.REGULARIZATION_GRID)
@pytest.mark.parametrize("arm", v.ARMS)
def test_same_source_only_through_explicit_inner_fold(regularization, arm):
    kw, arrays, _, _, _ = inner_inputs(arm, regularization)
    model = v.fit_kernel_decoder(**kw)
    assert model.training_source == model.held_source
    assert model.validation_kind == "inner_same_source_target"
    assert model.outer_fold_id == kw["fold"].outer_fold_id
    restored = restore(model, kw)
    np.testing.assert_array_equal(restored.dual_coefficients, model.dual_coefficients)
    saved = v.decoder_arrays(model)
    assert saved["regularization"].dtype == np.float64
    assert saved["regularization"].item() == regularization
    assert saved["validation_kind"].item() == "inner_same_source_target"
    assert saved["outer_fold_id"].item() == kw["fold"].outer_fold_id


@pytest.mark.parametrize("change", ["ordinary_same_source", "inner_cross_source", "missing_outer", "unknown_subclass"])
def test_false_inner_or_outer_scope_refused_before_array_reads(change):
    class DoNotRead:
        def __array__(self, *args, **kwargs): raise AssertionError("Unadmitted expression read")
    kw, _, _, _, _ = inner_inputs()
    inner = kw["fold"]
    if change == "ordinary_same_source":
        kw["fold"] = v3.JointFold(**{name: getattr(inner, name) for name in v3.JointFold.__dataclass_fields__})
    elif change == "inner_cross_source": kw["fold"] = replace(inner, held_source="J")
    elif change == "missing_outer": kw["fold"] = replace(inner, outer_fold_id="")
    else:
        class UnknownInner(v.InnerTargetFold): pass
        kw["fold"] = UnknownInner(**{name: getattr(inner, name) for name in inner.__dataclass_fields__})
    kw["context"] = kw["response"] = DoNotRead()
    with pytest.raises(ValueError): v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("mutation", ["rows", "duplicate_index", "negative_index", "boolean_index", "eval_outside", "other_overlap", "source", "target", "roster_duplicate"])
def test_group_and_target_admission_precedes_arrays(mutation):
    class DoNotRead:
        def __array__(self, *args, **kwargs): raise AssertionError("Unadmitted expression read")
    kw, _, _, _, _ = inputs(); fold = kw["fold"]
    if mutation == "rows": kw["fold"] = replace(fold, fit_indices=fold.fit_indices[:-1])
    elif mutation == "duplicate_index": kw["fold"] = replace(fold, fit_indices=(fold.fit_indices[0],)*len(fold.fit_indices))
    elif mutation == "negative_index": kw["fold"] = replace(fold, fit_indices=(-1, *fold.fit_indices[1:]))
    elif mutation == "boolean_index": kw["fold"] = replace(fold, fit_indices=(True, *fold.fit_indices[1:]))
    elif mutation == "eval_outside": kw["fold"] = replace(fold, evaluation_indices=fold.fit_indices)
    elif mutation == "other_overlap": kw["fold"] = replace(fold, other_excluded_indices=fold.fit_indices)
    elif mutation == "source": kw["sources"] = (fold.held_source,)*len(kw["sources"])
    elif mutation == "target": kw["targets"] = (fold.held_targets[0], *kw["targets"][1:])
    else: kw["fold"] = replace(fold, fit_targets=("C", "C", "D"))
    kw["context"] = kw["response"] = DoNotRead()
    with pytest.raises(ValueError): v.fit_kernel_decoder(**kw)


def test_overflowed_variance_cannot_silently_collapse_context():
    kw, _, _, _, _ = inputs("context")
    kw["context"] = np.array([[-1e200], [1e200], [-1e200], [1e200]])
    with pytest.raises(ValueError, match="mean/scale"):
        v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("bad_mean,bad_scale", [(np.inf, 1.), (np.nan, 1.), (0., np.inf), (0., np.nan), (0., 0.), (0., .01)])
def test_nonfinite_or_invalid_moments_are_rejected(monkeypatch, bad_mean, bad_scale):
    kw, _, _, _, _ = inputs("context")
    width = kw["context"].shape[1]
    monkeypatch.setattr(v, "weighted_standardize", lambda *args: (np.full(width, bad_mean), np.full(width, bad_scale)))
    with pytest.raises(ValueError, match="mean/scale"):
        v.fit_kernel_decoder(**kw)


@pytest.mark.parametrize("regularization", v.REGULARIZATION_GRID)
def test_training_diagnostics_lambda_bound_and_finite(regularization):
    kw, _, _, _, _ = inputs(regularization=regularization)
    model = v.fit_kernel_decoder(**kw)
    report = v.training_diagnostics(model, kw["response"], regularization=regularization)
    assert all(np.isfinite(x) and x >= 0 for x in report.values())
    wrong = next(x for x in v.REGULARIZATION_GRID if x != regularization)
    with pytest.raises(ValueError, match="regularization"):
        v.training_diagnostics(model, kw["response"], regularization=wrong)
    with pytest.raises(ValueError, match="differs"):
        v.training_diagnostics(model, kw["response"]+.1, regularization=regularization)


def test_less_regularization_increases_training_effective_df():
    kw, _, _, _, _ = inputs()
    values = []
    for regularization in v.REGULARIZATION_GRID:
        model = v.fit_kernel_decoder(**{**kw, "regularization": regularization})
        values.append(v.training_diagnostics(model, kw["response"], regularization=regularization)["effective_df"])
    assert values[0] > values[1] > values[2]


@pytest.mark.parametrize("kind", ["outer", "inner"])
@pytest.mark.parametrize("regularization", v.REGULARIZATION_GRID)
def test_restore_does_not_solve_or_refit(monkeypatch, kind, regularization):
    kw, _, _, _, _ = (inner_inputs if kind == "inner" else inputs)(regularization=regularization)
    model = v.fit_kernel_decoder(**kw)
    def forbidden(*args, **kwargs): raise AssertionError("Restoration may not solve/refit")
    monkeypatch.setattr(np.linalg, "solve", forbidden)
    monkeypatch.setattr(v, "fit_kernel_decoder", forbidden)
    restored = restore(model, kw)
    np.testing.assert_array_equal(restored.dual_coefficients, model.dual_coefficients)


def test_restore_lambda_mismatch_fails_before_expression_read():
    class DoNotRead:
        def __array__(self, *args, **kwargs): raise AssertionError("Expression read before lambda binding")
    kw, _, _, _, _ = inputs(regularization=.1); model = v.fit_kernel_decoder(**kw)
    with pytest.raises(ValueError, match="regularization"):
        restore(model, kw, regularization=1., context=DoNotRead(), response=DoNotRead())


@pytest.mark.parametrize("change", ["outer_id", "kind", "outer_fold"])
def test_restore_cannot_confuse_inner_and_outer_scope(change):
    kw, _, _, _, _ = inner_inputs(); model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    if change == "outer_id": saved["outer_fold_id"] = np.asarray("wrong_outer")
    elif change == "kind": saved["validation_kind"] = np.asarray("outer_joint_target_source")
    else:
        fold = kw["fold"]
        kw["fold"] = v3.JointFold(**{**{name: getattr(fold, name) for name in v3.JointFold.__dataclass_fields__}, "held_source": "J"})
    with pytest.raises(ValueError, match="mismatch"):
        restore(model, kw, saved=saved)


def test_forged_saved_lambda_with_old_coefficients_fails_dual_residual():
    kw, _, _, _, _ = inputs(regularization=1.); model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    saved["regularization"] = np.asarray(.1, dtype=np.float64)
    with pytest.raises(ValueError, match="residual"):
        restore(model, kw, saved=saved, regularization=.1)


@pytest.mark.parametrize("key,dtype", [("regularization", np.int64), ("context_mean", np.int64),
    ("context_scale", np.float32), ("training_target_features", np.float32), ("dual_coefficients", np.float32),
    ("training_weights", object), ("feature_signature_json", object), ("validation_kind", "S80")])
def test_unsafe_or_value_compatible_wrong_dtypes_rejected(key, dtype):
    kw, _, _, _, _ = inputs(regularization=1.)
    # Integral means make integer-cast context metadata numerically identical,
    # demonstrating why value-only comparison in v5 was insufficient.
    kw["context"] = np.ones_like(kw["context"])
    model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    saved[key] = saved[key].astype(dtype)
    with pytest.raises(ValueError, match="dtype|float64|Unicode"):
        restore(model, kw, saved=saved)


def test_wider_unicode_metadata_keeps_same_semantics():
    kw, _, _, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    saved["training_response_sha256"] = saved["training_response_sha256"].astype("U128")
    assert restore(model, kw, saved=saved).regularization == model.regularization


def test_model_state_is_immutable_and_does_not_alias_input_arrays():
    kw, _, _, _, _ = inputs(); model = v.fit_kernel_decoder(**kw)
    saved = {k: x.copy() for k, x in v.decoder_arrays(model).items()}
    kw["context"] += 100; kw["response"] += 200
    for key, value in saved.items(): np.testing.assert_array_equal(v.decoder_arrays(model)[key], value)
    for value in (model.context_mean, model.context_scale, model.training_context_features,
                  model.training_target_features, model.training_weights, model.dual_coefficients):
        assert not value.flags.writeable
    with pytest.raises(FrozenInstanceError): model.regularization = .1


@pytest.mark.parametrize("regularization", v.REGULARIZATION_GRID)
def test_zero_and_ntc_behavior_unchanged_for_every_lambda(regularization):
    kw, arrays, _, _, _ = inputs("control", regularization)
    model = v.fit_kernel_decoder(**kw)
    aligned = v3._subset(kw["aligned"], ("A", "B"))
    np.testing.assert_array_equal(model.predict_delta(aligned, arrays["context"][:2], is_ntc=np.array([False, True])),
                                  np.zeros((2, kw["response"].shape[1])))
    kw["arm"] = "true"; model = v.fit_kernel_decoder(**kw)
    result = model.predict_delta(aligned, arrays["context"][:2], is_ntc=np.array([False, True]))
    np.testing.assert_array_equal(result[1], np.zeros(kw["response"].shape[1]))
