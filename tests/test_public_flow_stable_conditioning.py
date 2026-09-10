"""Synthetic feature-only regressions; never open expression or retrain models."""
from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_flow_conditioning import (FeatureTable, align_target_features,
                                     ablate_target_features, fit_target_transform)
from public_flow_stable_conditioning import METHOD, fit_stable_target_transform


def align(values, ids=None, name="go"):
    values = np.asarray(values, dtype=np.float64)
    ids = tuple(ids) if ids is not None else tuple(f"t{i}" for i in range(len(values)))
    table = FeatureTable(name, ids, tuple(f"g{i}" for i in range(values.shape[1])),
                         values, "synthetic", "a" * 64)
    return align_target_features(ids, [table])


@pytest.mark.parametrize("count", [3, 9, 10, 13, 16, 100])
@pytest.mark.parametrize("scale", [True, False])
def test_repeated_fractional_constants_are_exact_zero(count, scale):
    values = np.tile([1/13, 2/13, 3/13, .1, .2, .3], (count, 1))
    aligned = align(values)
    fitted = fit_stable_target_transform(aligned, train_ids=aligned.target_ids,
                                         fold_id="training", scale=scale)
    np.testing.assert_array_equal(fitted.transform(aligned), np.zeros_like(values))
    np.testing.assert_array_equal(fitted.mean, values[0])
    np.testing.assert_array_equal(fitted.scale, np.ones(6))
    assert fitted.exact_constant_columns == tuple(range(6))


def test_regression_reproduces_original_roundoff_without_changing_legacy():
    aligned = align(np.tile([1/13, 2/13, 3/13, .1, .2, .3], (13, 1)))
    old = fit_target_transform(aligned, train_ids=aligned.target_ids, fold_id="old")
    assert np.any(old.transform(aligned) != 0)
    new = fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="new")
    np.testing.assert_array_equal(new.transform(aligned), np.zeros_like(aligned.values))


def test_large_exact_constants_do_not_overflow_mean_sum():
    values = np.tile([1e308, -1e308, 0.0], (13, 1))
    aligned = align(values)
    with np.errstate(over="raise", invalid="raise"):
        fitted = fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="large")
        np.testing.assert_array_equal(fitted.transform(aligned), np.zeros_like(values))


@pytest.mark.parametrize("scale", [True, False])
def test_variable_columns_preserve_legacy_moments(scale):
    values = np.array([[.1, 0, 2], [.1, 1, 3], [.1, 0, 4], [.1, 1, 8]])
    aligned = align(values)
    old = fit_target_transform(aligned, train_ids=aligned.target_ids, fold_id="fold", scale=scale)
    new = fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="fold", scale=scale)
    np.testing.assert_array_equal(new.mean[1:], old.mean[1:])
    np.testing.assert_array_equal(new.scale[1:], old.scale[1:])
    np.testing.assert_array_equal(new.transform(aligned)[:, 1:], old.transform(aligned)[:, 1:])
    assert new.exact_constant_columns == (0,)


def test_constant_training_column_can_differ_in_heldout_features():
    aligned = align([[.1, 0], [.1, 1], [5, 100]], ("train1", "train2", "heldout"))
    fitted = fit_stable_target_transform(aligned, train_ids=("train1", "train2"), fold_id="fold")
    np.testing.assert_array_equal(fitted.transform(aligned)[:2, 0], [0, 0])
    assert fitted.transform(aligned)[2, 0] == 4.9
    assert fitted.mean[0] == .1
    assert fitted.scale[0] == 1


def test_heldout_values_never_affect_fitted_statistics():
    first = align([[.1, 0], [.1, 1], [5, 100]], ("A", "B", "heldout"))
    second = align([[.1, 0], [.1, 1], [-1e200, 1e200]], ("A", "B", "heldout"))
    fits = [fit_stable_target_transform(a, train_ids=("A", "B"), fold_id="fold") for a in (first, second)]
    np.testing.assert_array_equal(fits[0].mean, fits[1].mean)
    np.testing.assert_array_equal(fits[0].scale, fits[1].scale)
    assert fits[0].exact_constant_columns == fits[1].exact_constant_columns


def test_multiple_modality_missingness_preserved():
    go = FeatureTable("go", ("A", "B"), ("go1", "go2"), np.array([[.1, 1], [.1, 0]]), "s", "a" * 64)
    gene = FeatureTable("other", ("B", "C"), ("f1",), np.array([[.3], [.3]]), "s", "b" * 64)
    aligned = align_target_features(("A", "B", "C", "missing"), [go, gene])
    original_mask = aligned.present.copy()
    fitted = fit_stable_target_transform(aligned, train_ids=("A", "B", "C"), fold_id="fold")
    transformed = fitted.transform(aligned)
    np.testing.assert_array_equal(aligned.present, original_mask)
    np.testing.assert_array_equal(transformed[3], [0, 0, 0])
    np.testing.assert_array_equal(transformed[2, :2], [0, 0])
    assert transformed[0, 2] == 0
    assert fitted.observed_training_rows == (2, 2)
    assert fitted.exact_constant_columns == (0, 2)


@pytest.mark.parametrize("mode", ["constant", "shuffle"])
def test_ablation_source_signature_and_masks_preserved(mode):
    aligned = align([[1, 0], [0, 1], [1, 1]])
    ablated = ablate_target_features(aligned, mode=mode, seed=3,
                                     train_ids=aligned.target_ids, fold_id="fold")
    fitted = fit_stable_target_transform(ablated, train_ids=aligned.target_ids, fold_id="fold")
    assert fitted.signature == ablated.signature
    if mode == "constant":
        np.testing.assert_array_equal(fitted.transform(ablated), np.zeros_like(aligned.values))
    with pytest.raises(ValueError, match="changed"):
        fitted.transform(aligned)


@pytest.mark.parametrize("ids,excluded", [(('A', 'A'), ()), (('A', 'missing'), ()), (('A',), ('A',)), ((), ())])
def test_invalid_training_admission_rejected(ids, excluded):
    aligned = align([[.1], [.1]], ("A", "B"))
    with pytest.raises(ValueError):
        fit_stable_target_transform(aligned, train_ids=ids, excluded_ids=excluded, fold_id="fold")


def test_no_observed_modality_training_rows_rejected():
    table = FeatureTable("go", ("heldout",), ("g1",), np.ones((1, 1)), "s", "a" * 64)
    aligned = align_target_features(("train", "heldout"), [table])
    with pytest.raises(ValueError, match="No observed training features"):
        fit_stable_target_transform(aligned, train_ids=("train",), fold_id="fold")


@pytest.mark.parametrize("pca", [0, 1, True, "1"])
def test_pca_explicitly_rejected(pca):
    aligned = align([[1], [1]])
    with pytest.raises(ValueError, match="PCA"):
        fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="fold", pca_components=pca)


@pytest.mark.parametrize("scale", [None, 0, 1, "true"])
def test_nonboolean_scale_rejected(scale):
    aligned = align([[1], [1]])
    with pytest.raises(ValueError, match="boolean"):
        fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="fold", scale=scale)


def test_provenance_names_new_opt_in_method_and_preserves_fold():
    aligned = align([[.1, 0], [.1, 1]])
    fitted = fit_stable_target_transform(aligned, train_ids=aligned.target_ids, fold_id="registered_fold")
    info = fitted.provenance()
    assert info["standardization_method"] == METHOD
    assert info["fold_id"] == "registered_fold"
    assert info["exact_constant_columns"] == [0]
    assert info["pca_supported"] is False
    assert fitted.mean.flags.writeable is False
    assert fitted.scale.flags.writeable is False


def test_input_is_not_mutated():
    aligned = align([[.1, 0], [.1, 1], [10, 2]])
    original = aligned.values.copy()
    fit_stable_target_transform(aligned, train_ids=aligned.target_ids[:2], fold_id="fold")
    np.testing.assert_array_equal(aligned.values, original)
