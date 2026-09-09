"""Synthetic-only tests: conditioning screens cannot authorize submissions."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import validate_lingshu_conditioning as screen
from diagnose_lingshu_training_signal import fit_predict
from lingshu_conditioning_adapter import AdapterSpec, TrainingBasis, matrix


def signal(seed=17, count=60):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(count, 3))
    weights = np.asarray([[1.0, -0.3, 0.7, 0.1], [0.2, 0.8, -0.4, 0.5], [-0.6, 0.4, 0.2, 0.9]])
    effects = features @ weights + np.asarray([0.7, -0.2, 0.4, 0.1])
    return features, effects


def full_spec(**kwargs):
    values = dict(input_rank=3, effect_rank=3, penalty=1e-8, mean_gain=1.0, residual_gain=1.0)
    values.update(kwargs)
    return AdapterSpec(**values)


@pytest.fixture
def small_grid(monkeypatch):
    monkeypatch.setattr(screen, "INPUT_RANKS", (3,))
    monkeypatch.setattr(screen, "EFFECT_RANKS", (3,))
    monkeypatch.setattr(screen, "PENALTIES", (1e-6,))
    monkeypatch.setattr(screen, "RESIDUAL_GAINS", (0.0, 1.0))
    monkeypatch.setattr(screen, "SHUFFLE_SEEDS", (71, 73))


def test_recovers_heldout_synthetic_feature_signal():
    features, effects = signal()
    adapter = TrainingBasis(features[:48], effects[:48]).fit(full_spec())
    np.testing.assert_allclose(adapter.predict_effect(features[48:]), effects[48:], atol=1e-6)
    assert adapter.condition(features[48:]).shape == (12, 3)


def test_feature_scaling_has_unit_mean_squared_row_norm():
    features, effects = signal()
    basis = TrainingBasis(features, effects)
    scaled = (features - basis.feature_center) / basis.feature_scale
    assert np.mean(np.sum(scaled ** 2, axis=1)) == pytest.approx(1.0)
    assert np.mean(scaled ** 2) == pytest.approx(1 / features.shape[1])


@pytest.mark.parametrize("penalty", [0.001, 0.1, 1.0, 10.0])
def test_full_rank_adapter_matches_centered_linear_kernel_ridge(penalty):
    rng = np.random.default_rng(19)
    features = rng.normal(size=(48, 4))
    effects = rng.normal(size=(48, 3))
    query = rng.normal(size=(7, 4))
    adapter = TrainingBasis(features, effects).fit(full_spec(input_rank=4, penalty=penalty))
    expected, expected_train = fit_predict(features, effects, query, "linear", penalty)
    np.testing.assert_allclose(adapter.predict_effect(query), expected, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(adapter.predict_effect(features), expected_train, rtol=1e-10, atol=1e-10)


def test_predictions_do_not_depend_on_query_batch_or_refit_training_transforms():
    features, effects = signal()
    adapter = TrainingBasis(features[:48], effects[:48]).fit(full_spec())
    original_center = adapter.feature_center.copy()
    original_effect_center = adapter.effect_center.copy()
    expected = adapter.predict_effect(features[48:49])
    queries = np.concatenate([features[48:49], np.full((5, 3), 1e9)])
    np.testing.assert_allclose(adapter.predict_effect(queries)[:1], expected, atol=1e-12)
    np.testing.assert_array_equal(adapter.feature_center, original_center)
    np.testing.assert_array_equal(adapter.effect_center, original_effect_center)
    np.testing.assert_allclose(original_center, features[:48].mean(axis=0))
    np.testing.assert_allclose(original_effect_center, effects[:48].mean(axis=0))


def test_train_effect_center_is_separate_from_residual_conditioning():
    features, effects = signal()
    shift = np.asarray([3.0, -7.0, 4.0, 2.0])
    first = TrainingBasis(features[:48], effects[:48]).fit(full_spec())
    second = TrainingBasis(features[:48], effects[:48] + shift).fit(full_spec())
    np.testing.assert_allclose(second.predict_effect(features[48:]) - shift,
                               first.predict_effect(features[48:]), atol=1e-10)
    np.testing.assert_allclose(first.condition(features[:48]).mean(axis=0), 0, atol=1e-12)


@pytest.mark.parametrize("mean_gain", [0.0, 0.5, 1.0])
def test_zero_residual_gain_exactly_matches_train_only_constant(mean_gain):
    features, effects = signal()
    adapter = TrainingBasis(features[:48], effects[:48]).fit(
        full_spec(mean_gain=mean_gain, residual_gain=0.0))
    prediction = adapter.predict_effect(features[48:])
    expected = np.broadcast_to(mean_gain * effects[:48].mean(axis=0), prediction.shape)
    np.testing.assert_array_equal(prediction, expected)
    np.testing.assert_array_equal(adapter.condition(features[48:]), 0)


def test_constant_features_cannot_fabricate_target_signal():
    _, effects = signal()
    adapter = TrainingBasis(np.ones((60, 7)), effects).fit(full_spec(input_rank=7))
    assert adapter.feature_basis.shape == (7, 0)
    prediction = adapter.predict_effect(np.full((4, 7), 100.0))
    np.testing.assert_allclose(prediction, np.broadcast_to(effects.mean(axis=0), prediction.shape))
    np.testing.assert_array_equal(adapter.coefficients, 0)


@pytest.mark.parametrize("effect_value", [0.0, 2.0])
def test_degenerate_effect_basis_remains_finite_and_constant(effect_value):
    features, effects = signal()
    effects[:] = effect_value
    adapter = TrainingBasis(features, effects).fit(full_spec())
    assert adapter.effect_basis.shape == (4, 0)
    assert adapter.condition(features[:3]).shape == (3, 0)
    np.testing.assert_array_equal(adapter.predict_effect(features[:3]), effect_value)


def test_rank_is_capped_by_training_samples_and_nonzero_singular_directions():
    features = np.arange(5.0)[:, None] * np.asarray([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])
    effects = features[:, :3] + 2.0
    basis = TrainingBasis(features, effects, max_input_rank=100, max_effect_rank=100)
    adapter = basis.fit(full_spec(input_rank=100, effect_rank=100))
    assert adapter.feature_basis.shape == (6, 1)
    assert adapter.effect_basis.shape == (3, 1)
    assert np.isfinite(adapter.predict_effect(features)).all()


def test_stronger_ridge_shrinks_conditioning_coefficients():
    features, effects = signal()
    basis = TrainingBasis(features, effects)
    weak = basis.fit(full_spec(penalty=1e-6))
    strong = basis.fit(full_spec(penalty=100.0))
    assert np.linalg.norm(strong.coefficients) < np.linalg.norm(weak.coefficients) / 10
    assert np.linalg.norm(strong.condition(features)) < np.linalg.norm(weak.condition(features)) / 10


@pytest.mark.parametrize("kwargs", [
    {"input_rank": 0}, {"effect_rank": -1}, {"penalty": 0.0}, {"penalty": -1.0},
    {"penalty": np.nan}, {"penalty": np.inf}, {"mean_gain": -0.1},
    {"mean_gain": np.nan}, {"residual_gain": 1.1},
])
def test_invalid_adapter_specs_are_rejected(kwargs):
    with pytest.raises(ValueError):
        full_spec(**kwargs).validate()


@pytest.mark.parametrize("value", [[], [1, 2], [[np.inf]], [[np.nan]], np.empty((0, 2))])
def test_invalid_matrices_are_rejected(value):
    with pytest.raises(ValueError):
        matrix(value)


def test_training_alignment_sample_count_and_query_dimensions_are_checked():
    features, effects = signal()
    with pytest.raises(ValueError, match="aligned"):
        TrainingBasis(features[:5], effects[:4])
    with pytest.raises(ValueError, match="three"):
        TrainingBasis(features[:2], effects[:2])
    adapter = TrainingBasis(features, effects).fit(full_spec())
    with pytest.raises(ValueError, match="dimension"):
        adapter.predict_effect(np.zeros((3, 4)))


def test_research_export_is_pickle_free_exclusive_and_not_submission_permission(tmp_path):
    features, effects = signal()
    adapter = TrainingBasis(features, effects).fit(full_spec())
    path = tmp_path / "adapter.npz"
    metadata = {"gene_axis": ["g1", "g2", "g3", "g4"], "cache_sha256": "synthetic-only"}
    adapter.save_research_only(path, metadata=metadata)
    contents = path.read_bytes()
    with np.load(path, allow_pickle=False) as archive:
        assert archive["schema"].item() == "vcc-lingshu-low-rank-conditioning-research-v4"
        assert archive["submission_allowed"].item() is False
        assert archive["independent_test"].item() is False
        assert archive["requires_authenticated_sidecars"].item() is True
        assert json.loads(archive["metadata"].item()) == metadata
        assert json.loads(archive["spec"].item())["penalty"] == adapter.spec.penalty
        assert all(archive[key].dtype.kind != "O" for key in archive.files)
        np.testing.assert_array_equal(archive["coefficients"], adapter.coefficients)
    with pytest.raises(FileExistsError):
        adapter.save_research_only(path)
    assert path.read_bytes() == contents


def test_json_receipts_refuse_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    screen.write_json(path, {"submission_allowed": False})
    with pytest.raises(FileExistsError):
        screen.write_json(path, {"submission_allowed": True})
    assert json.loads(path.read_text()) == {"submission_allowed": False}


def test_training_loader_allowlist_and_matched_control_centering():
    targets = np.repeat(np.asarray([f"g{i:02d}" for i in range(15)]), 2)
    control = np.arange(90.0).reshape(30, 3)
    effects = np.repeat(np.arange(15.0)[:, None], 3, axis=1)
    values = {"train_targets": targets, "train_contexts": np.asarray(["K562"] * 30),
              "train_x": control + np.repeat(effects, 2, axis=0), "train_control": control}

    class GuardedArchive:
        def __init__(self):
            self.read = []

        def __getitem__(self, key):
            assert key in screen.TRAIN_KEYS, "Validation or challenge truth was requested"
            self.read.append(key)
            return values[key]

    archive = GuardedArchive()
    names, loaded_effects, counts = screen.training_effects(archive)
    assert tuple(archive.read) == screen.TRAIN_KEYS
    np.testing.assert_array_equal(names, targets[::2])
    np.testing.assert_array_equal(loaded_effects, effects)
    assert counts == [2] * 15
    values["train_contexts"][0] = "RPE1"
    with pytest.raises(ValueError, match="K562"):
        screen.training_effects(archive)


def test_shuffled_arms_are_exact_bijections_of_same_target_feature_pool(small_grid):
    features, _ = signal(count=18)
    names = np.asarray([f"g{i}" for i in range(18)])
    arms, mappings = screen.feature_arms(features, names)
    np.testing.assert_array_equal(arms["true"], features)
    original = dict(zip(names, features, strict=True))
    assert set(arms) == {"true", "shuffled_71", "shuffled_73"}
    for arm, mapped_names in mappings.items():
        assert set(mapped_names) == set(names)
        assert set(mapped_names.values()) == set(names)
        assert len(set(mapped_names.values())) == len(names)
        for index, name in enumerate(names):
            np.testing.assert_array_equal(arms[arm][index], original[mapped_names[name]])
    again, again_mapping = screen.feature_arms(features, names)
    assert again_mapping == mappings
    for arm in arms:
        np.testing.assert_array_equal(arms[arm], again[arm])


def test_candidates_include_exactly_one_constant_arm_and_simple_tie_breaking(small_grid):
    specs = list(screen.candidates(0.5))
    assert len([spec for spec in specs if spec.residual_gain == 0]) == 1
    assert all(spec.mean_gain == 0.5 for spec in specs)
    assert screen.complexity(full_spec(residual_gain=0)) < screen.complexity(full_spec())
    assert screen.complexity(full_spec(penalty=10)) < screen.complexity(full_spec(penalty=1))


def test_constant_and_adapter_selection_prefers_zero_for_no_effect_signal(small_grid):
    features, effects = signal(count=18)
    effects[:] = 0
    splits = screen.folds(len(features), 3, 55)
    gain, scores = screen.select_constant(effects, splits)
    assert gain == 0.0
    assert all(score == 0 for score in scores.values())
    spec, candidate_scores = screen.select_adapter(features, effects, splits, gain)
    assert spec.residual_gain == 0.0
    assert all(record["inner_mse"] == 0 for record in candidate_scores)


def test_inner_selection_fits_each_basis_on_inner_training_targets_only(small_grid, monkeypatch):
    features, effects = signal(count=18)
    splits = screen.folds(18, 3, 55)
    observed = []

    def tracked_basis(train_features, train_effects):
        observed.append((train_features.copy(), train_effects.copy()))
        return TrainingBasis(train_features, train_effects)

    monkeypatch.setattr(screen, "TrainingBasis", tracked_basis)
    screen.select_adapter(features, effects, splits, mean_gain=1.0)
    assert len(observed) == len(splits)
    for heldout, (actual_features, actual_effects) in zip(splits, observed, strict=True):
        train = np.setdiff1d(np.arange(len(features)), heldout)
        np.testing.assert_array_equal(actual_features, features[train])
        np.testing.assert_array_equal(actual_effects, effects[train])


def test_adapter_selection_scores_all_effect_genes_not_only_retained_pcs(monkeypatch):
    features, _ = signal(count=18)
    effects = np.tile([0.0, 0.0, 4.0, 4.0], (18, 1))
    monkeypatch.setattr(screen, "candidates", lambda gain: [full_spec(mean_gain=0, residual_gain=0)])
    _, scores = screen.select_adapter(features, effects, screen.folds(18, 3, 55), 0)
    assert scores[0]["inner_mse"] == pytest.approx(8.0)


def test_repeated_losses_are_averaged_by_target_before_bootstrapping():
    baseline = np.full((2, 4), 10.0)
    errors = baseline + np.asarray([[0.0, 2.0, 4.0, 6.0], [-2.0, -4.0, -6.0, -8.0]])
    comparison = screen.paired_comparison(errors, baseline)
    assert comparison["bootstrap_unit"] == "target_after_averaging_repeated_out_of_fold_losses"
    assert comparison["mean_mse_difference"] == pytest.approx(-1.0)
    np.testing.assert_allclose(comparison["exploratory_target_bootstrap_95pct"], [-1.0, -1.0])
    np.testing.assert_allclose(comparison["repeat_mse_differences"], [3.0, -5.0])
    assert comparison["relative_improvement"] == pytest.approx(0.1)


def decision_errors():
    return {"true": np.ones((3, 20)), "constant": np.full((3, 20), 2.0),
            "zero": np.full((3, 20), 3.0),
            **{f"shuffled_{seed}": np.full((3, 20), 2.5) for seed in screen.SHUFFLE_SEEDS}}


def test_positive_screen_is_only_eligible_for_separate_pilot_never_submission():
    result = screen.decision(decision_errors())
    assert result["conditioning_screen_passed"]
    assert all(result["checks"].values())
    assert result["next_stage"] == "eligible_for_separate_context_distribution_pilot"
    assert result["submission_allowed"] is False
    assert result["official_public_quality_gates_changed"] is False


@pytest.mark.parametrize("baseline", ["constant", "zero", *[f"shuffled_{seed}" for seed in screen.SHUFFLE_SEEDS]])
def test_matching_any_required_control_blocks_screen(baseline):
    errors = decision_errors()
    errors[baseline] = errors["true"].copy()
    result = screen.decision(errors)
    assert not result["conditioning_screen_passed"]
    assert result["next_stage"] == "blocked_pending_better_public_conditioning_signal"
    assert result["submission_allowed"] is False


@pytest.mark.parametrize("missing", ["true", "constant", "zero", *[f"shuffled_{seed}" for seed in screen.SHUFFLE_SEEDS]])
def test_missing_required_control_is_rejected(missing):
    errors = decision_errors()
    del errors[missing]
    with pytest.raises(ValueError):
        screen.decision(errors)


@pytest.mark.parametrize("malformed", [np.full((3, 20), np.nan), np.full((3, 20), np.inf),
                                     np.full((3, 20), -1), np.ones((2, 20))])
def test_nonfinite_negative_or_misaligned_comparison_errors_are_rejected(malformed):
    errors = decision_errors()
    errors["constant"] = malformed
    with pytest.raises(ValueError):
        screen.decision(errors)


@pytest.mark.parametrize("shape", [(20,), (0, 20), (3, 1), (3, 20, 1)])
def test_invalid_repeat_target_error_shapes_are_rejected(shape):
    errors = {key: np.ones(shape) for key in decision_errors()}
    with pytest.raises(ValueError):
        screen.decision(errors)


def test_unregistered_extra_control_is_rejected():
    errors = decision_errors()
    errors["unregistered"] = np.ones((3, 20))
    with pytest.raises(ValueError):
        screen.decision(errors)


def test_less_than_one_percent_gain_is_not_promoted():
    errors = decision_errors()
    errors["true"][:] = 1.99
    result = screen.decision(errors)
    assert not result["checks"]["at_least_one_percent_better_than_optimized_constant"]
    assert not result["conditioning_screen_passed"]


def test_improvement_on_average_cannot_hide_a_failed_repeat():
    errors = decision_errors()
    errors["true"][0] = 2.1
    assert errors["true"].mean() < errors["constant"].mean()
    result = screen.decision(errors)
    assert not result["checks"]["better_than_constant_in_each_repeat"]
    assert not result["conditioning_screen_passed"]


def test_small_nested_cv_recovers_signal_with_disjoint_training_and_holdouts(small_grid):
    features, effects = signal(count=48)
    names = np.asarray([f"g{i:02d}" for i in range(48)])
    seeds = (31, 37)
    result, errors = screen.evaluate(features, effects, names, seeds=seeds, outer_folds=3)
    assert result["decision"]["conditioning_screen_passed"]
    assert result["decision"]["submission_allowed"] is False
    assert result["mse"]["true"] < result["mse"]["constant"] * 0.1
    assert len(result["folds"]) == len(seeds) * 3 * 3
    for values in errors.values():
        assert values.shape == (len(seeds), len(names))
        assert np.isfinite(values).all()
        assert (values >= 0).all()
    for seed in seeds:
        for arm in ("true", "shuffled_71", "shuffled_73"):
            records = [record for record in result["folds"]
                       if record["repeat_seed"] == seed and record["arm"] == arm]
            heldouts = []
            for record in records:
                train, heldout = set(record["train_targets"]), set(record["heldout_targets"])
                assert train.isdisjoint(heldout)
                assert train | heldout == set(names)
                inner = [name for part in record["inner_fold_targets"] for name in part]
                assert set(inner) == train
                assert len(inner) == len(set(inner))
                heldouts.extend(record["heldout_targets"])
            assert set(heldouts) == set(names)
            assert len(heldouts) == len(names)


def test_login_host_guard_fails_before_hashing_or_loading_data(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["validate_lingshu_conditioning.py", "--cache", "unused.npz",
                                    "--embeddings", "unused-embeddings.npz", "--embedding-receipt",
                                    "unused.json", "--output-dir", str(tmp_path / "not-created")])
    monkeypatch.setattr(screen.socket, "getfqdn", lambda: "login.example.edu")
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)

    def forbidden_read(_):
        pytest.fail("The login-node guard must run before file hashing")

    monkeypatch.setattr(screen, "digest", forbidden_read)
    with pytest.raises(RuntimeError, match="not a login node"):
        screen.main()
    assert not (tmp_path / "not-created").exists()
