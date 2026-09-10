"""Synthetic-only checks; no datasets, networks, GPUs, or model weights."""
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from public_flow_conditioning import (  # noqa: E402
    FeatureTable, ablate_target_features, align_target_features,
    fit_target_transform, summarize_matched_controls,
)


def table(name="genept", targets=("B", "A", "C"), values=None, source="a" * 64):
    if values is None:
        values = np.array([[4., 8.], [2., 4.], [100., 200.]])
    return FeatureTable(name, targets, tuple(f"f{i}" for i in range(np.shape(values)[1])),
                        values, f"synthetic:{name}:v1", source)


def fixture_tables():
    return (table(), table("go", ("C", "A"), np.array([[1., 0.], [0., 0.]])))


def test_alignment_is_keyed_and_missing_not_confused_with_known_zero():
    aligned = align_target_features(("A", "B", "C", "UNKNOWN"), fixture_tables())
    np.testing.assert_array_equal(aligned.values,
                                  [[2, 4, 0, 0], [4, 8, 0, 0], [100, 200, 1, 0], [0, 0, 0, 0]])
    np.testing.assert_array_equal(aligned.present, [[1, 1], [1, 0], [1, 1], [0, 0]])
    np.testing.assert_array_equal(aligned.unknown_target, [False, False, False, True])
    assert aligned.modality_names == ("genept", "go")
    assert aligned.widths == (2, 2)


def test_empty_modality_has_explicit_false_mask_but_cannot_fit_unseen_statistics():
    empty = table("go", (), np.zeros((0, 2)))
    aligned = align_target_features(("A", "B"), [table(), empty])
    assert not aligned.present[:, 1].any()
    with pytest.raises(ValueError, match="No observed training"):
        fit_target_transform(aligned, train_ids=("A", "B"), fold_id="outer-0")


@pytest.mark.parametrize("changes", [
    {"target_ids": ("A", "A", "C")}, {"feature_ids": ("f0", "f0")},
    {"name": ""}, {"source_id": ""}, {"source_sha256": "not-a-hash"},
    {"source_sha256": "A" * 64}, {"values": np.zeros((3, 3))},
    {"values": np.array([[1, 2], [3, np.nan], [5, 6]])},
    {"values": np.array([[1, 2], [3, np.inf], [5, 6]])},
    {"values": np.ones((3, 2), dtype=complex)},
    {"values": np.ones((3, 2), dtype=object)},
    {"values": np.ones((3, 2), dtype=bool)},
    {"target_ids": "ABC"}, {"target_ids": ("A", " B", "C")},
])
def test_bad_feature_tables_rejected(changes):
    with pytest.raises(ValueError):
        replace(table(), **changes)


@pytest.mark.parametrize("targets,tables", [
    (("A", "A"), (table(),)), ((), (table(),)), (("A",), ()),
    (("A",), (table(), table())), (("A",), (np.ones((2, 2)),)),
])
def test_alignment_rejects_ambiguous_axes(targets, tables):
    with pytest.raises(ValueError):
        align_target_features(targets, tables)


def test_input_arrays_are_copied_and_results_read_only():
    original = np.ones((3, 2))
    feature_table = table(values=original)
    original[:] = 100
    assert (feature_table.values == 1).all()
    with pytest.raises(ValueError):
        feature_table.values[0, 0] = 10
    aligned = align_target_features(("A", "B"), [feature_table])
    with pytest.raises(ValueError):
        aligned.present[0, 0] = False


def test_train_fold_statistics_do_not_include_outlier_or_missing_blocks():
    aligned = align_target_features(("A", "B", "C", "UNKNOWN"), fixture_tables())
    fitted = fit_target_transform(aligned, train_ids=("A", "B"), fold_id="outer-0",
                                  excluded_ids=("C", "UNKNOWN"))
    np.testing.assert_array_equal(fitted.mean, [3, 6, 0, 0])
    np.testing.assert_array_equal(fitted.scale, [1, 2, 1, 1])
    assert fitted.observed_training_rows == (2, 1)
    transformed = fitted.transform(aligned)
    np.testing.assert_array_equal(transformed[-1], 0)
    np.testing.assert_array_equal(transformed[1, 2:], 0)
    np.testing.assert_array_equal(transformed[:2, :2], [[-1, -1], [1, 1]])
    assert fitted.provenance()["fold_id"] == "outer-0"
    assert len(fitted.provenance()["training_ids_sha256"]) == 64


def test_holdout_query_batch_does_not_refit_transform():
    sources = fixture_tables()
    aligned = align_target_features(("A", "B", "C", "UNKNOWN"), sources)
    fitted = fit_target_transform(aligned, train_ids=("A", "B"), fold_id="outer-0")
    provenance = fitted.provenance()
    single = fitted.transform(align_target_features(("C",), sources))
    batch = fitted.transform(align_target_features(("UNKNOWN", "C", "A"), sources))
    np.testing.assert_array_equal(single[0], batch[1])
    assert fitted.provenance() == provenance


def test_pca_uses_only_training_rows_and_has_deterministic_signs(monkeypatch):
    aligned = align_target_features(("A", "B", "C", "UNKNOWN"), [table()])
    seen = []
    svd = np.linalg.svd

    def record(values, **kwargs):
        seen.append(values.copy())
        return svd(values, **kwargs)

    monkeypatch.setattr(np.linalg, "svd", record)
    fitted = fit_target_transform(aligned, train_ids=("A", "B"), fold_id="f", pca_components=1)
    assert len(seen) == 1 and seen[0].shape == (2, 2)
    np.testing.assert_array_equal(seen[0], [[-1, -1], [1, 1]])
    assert fitted.components[0, np.argmax(abs(fitted.components[0]))] >= 0
    np.testing.assert_array_equal(fitted.transform(aligned)[-1], 0)


def test_changed_heldout_features_cannot_change_fitted_statistics_or_pca():
    first = table()
    changed = replace(first, values=np.array([[4., 8.], [2., 4.], [-1e9, 5e9]]))
    fits = [fit_target_transform(align_target_features(("A", "B", "C"), [source]),
                                 train_ids=("A", "B"), fold_id="f", pca_components=1)
            for source in (first, changed)]
    for attribute in ("mean", "scale", "components"):
        np.testing.assert_array_equal(getattr(fits[0], attribute), getattr(fits[1], attribute))


@pytest.mark.parametrize("changes", [
    {"train_ids": ()}, {"train_ids": ("A", "A")}, {"train_ids": ("missing",)},
    {"excluded_ids": ("A",)}, {"fold_id": ""}, {"scale": 1},
    {"pca_components": 0}, {"pca_components": 2}, {"pca_components": True},
])
def test_invalid_fit_contract_rejected(changes):
    kwargs = dict(train_ids=("A", "B"), fold_id="f")
    kwargs.update(changes)
    with pytest.raises(ValueError):
        fit_target_transform(align_target_features(("A", "B", "C"), [table()]), **kwargs)


def test_transform_rejects_changed_source_layout_or_ablation():
    aligned = align_target_features(("A", "B", "C"), fixture_tables())
    fitted = fit_target_transform(aligned, train_ids=("A", "B"), fold_id="f")
    changes = [align_target_features(aligned.target_ids, fixture_tables()[::-1]),
               align_target_features(aligned.target_ids,
                                     [replace(table(), source_sha256="b" * 64), fixture_tables()[1]]),
               ablate_target_features(aligned, mode="zero")]
    for other in changes:
        with pytest.raises(ValueError, match="changed"):
            fitted.transform(other)


def test_scale_false_centers_without_rescaling():
    aligned = align_target_features(("A", "B"), [table()])
    fitted = fit_target_transform(aligned, train_ids=("A", "B"), fold_id="f", scale=False)
    np.testing.assert_array_equal(fitted.transform(aligned), [[-1, -2], [1, 2]])


def test_constant_and_zero_arms_keep_missingness_and_fit_constant_on_train_only():
    aligned = align_target_features(("A", "B", "C", "UNKNOWN"), fixture_tables())
    constant = ablate_target_features(aligned, mode="constant", train_ids=("A", "B"), fold_id="f")
    np.testing.assert_array_equal(constant.values, [[3, 6, 0, 0], [3, 6, 0, 0],
                                                  [3, 6, 0, 0], [0, 0, 0, 0]])
    zero = ablate_target_features(aligned, mode="zero")
    np.testing.assert_array_equal(zero.values, 0)
    np.testing.assert_array_equal(constant.present, aligned.present)
    np.testing.assert_array_equal(zero.present, aligned.present)
    for kwargs in ({}, {"train_ids": ("A", "B")}, {"train_ids": ("UNKNOWN",), "fold_id": "f"}):
        with pytest.raises(ValueError):
            ablate_target_features(aligned, mode="constant", **kwargs)


def test_shuffle_deterministic_joint_bijection_preserves_missingness_and_global_rng():
    ids = tuple(f"T{i}" for i in range(20))
    a = table(targets=ids, values=np.arange(40).reshape(20, 2))
    b = table("go", ids[:16], np.arange(32).reshape(16, 2) + 100)
    aligned = align_target_features(ids + ("UNKNOWN",), [a, b])
    np.random.seed(77)
    before = np.random.get_state()
    one = ablate_target_features(aligned, mode="shuffle", seed=12)
    two = ablate_target_features(aligned, mode="shuffle", seed=12)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    np.testing.assert_array_equal(one.values, two.values)
    assert not np.array_equal(one.values, aligned.values)
    np.testing.assert_array_equal(one.present, aligned.present)
    assert sorted(map(tuple, one.values)) == sorted(map(tuple, aligned.values))
    assert (one.values[:16, 2:] - one.values[:16, :2] == 100).all()
    np.testing.assert_array_equal(one.values[-1], 0)


def control_kwargs():
    return dict(cell_ids=("ctrl1", "ctrl2"), context_ids=("K562", "K562"),
                match_keys=("donor1|batch1", "donor1|batch1"), is_control=np.array([True, True]),
                requested_context="K562", requested_match_key="donor1|batch1",
                feature_ids=("G1", "G2"), source_id="synthetic:controls:v1",
                source_sha256="c" * 64, representation="counts")


def test_control_summary_population_statistics_and_provenance():
    summary = summarize_matched_controls([[1, 5], [3, 9]], **control_kwargs())
    np.testing.assert_array_equal(summary.values, [2, 7, 1, 2])
    assert summary.n_controls == 2
    assert summary.provenance["control_only"] is True
    assert summary.provenance["match_key"] == "donor1|batch1"
    assert summary.provenance["representation"] == "counts"
    assert "ctrl1" not in str(summary.provenance)
    mean = summarize_matched_controls([[1, 5], [3, 9]], include_std=False, **control_kwargs())
    np.testing.assert_array_equal(mean.values, [2, 7])
    with pytest.raises(ValueError):
        summary.values[0] = 10


class UnreadableValues:
    def __array__(self, *args, **kwargs):
        raise AssertionError("Expression accessed before the metadata firewall")


@pytest.mark.parametrize("changes", [
    {"is_control": [True, False]}, {"is_control": [False, False]},
    {"is_control": [1, 1]}, {"is_control": ["True", "True"]},
    {"is_control": [True]}, {"context_ids": ("K562", "RPE1")},
    {"requested_context": "RPE1"}, {"match_keys": ("donor1|batch1", "donor1|batch2")},
    {"requested_match_key": "donor2|batch1"}, {"cell_ids": ("duplicate", "duplicate")},
    {"cell_ids": ()}, {"feature_ids": ("G1", "G1")},
    {"source_id": ""}, {"source_sha256": "invalid"}, {"representation": ""},
])
def test_control_metadata_firewall_precedes_expression_access(changes):
    kwargs = control_kwargs()
    kwargs.update(changes)
    with pytest.raises(ValueError):
        summarize_matched_controls(UnreadableValues(), **kwargs)


@pytest.mark.parametrize("values", [np.zeros((3, 2)), np.zeros((2, 3)),
                                  [[1, np.nan], [2, 3]], [[1, 2], [3, np.inf]],
                                  np.zeros((2, 2), dtype=object)])
def test_control_expression_shape_and_finite_checks(values):
    with pytest.raises(ValueError):
        summarize_matched_controls(values, **control_kwargs())
