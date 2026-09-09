"""Synthetic CPU-only tests of the biological conditioning research screen."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import pytest


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import lingshu_conditioning_adapter as adapter_module  # noqa: E402
from lingshu_conditioning_adapter import AdapterSpec  # noqa: E402
import train_biological_conditioning_v5 as screen  # noqa: E402


def synthetic(n=18):
    rng = np.random.default_rng(821)
    esm2 = rng.normal(size=(n, 5))
    language = rng.normal(size=(n, 9))
    effects = esm2 @ rng.normal(size=(5, 4)) + 0.1 * rng.normal(size=(n, 4))
    names = np.asarray([f"SYNTHETIC_{index:03d}" for index in range(n)])
    return esm2, language, effects, names


def small_grid(mean_gain):
    yield AdapterSpec(2, 2, 10.0, mean_gain, 0.0)
    yield AdapterSpec(2, 2, 1.0, mean_gain, 0.25)


def valid_errors():
    errors = {"constant": np.ones((3, 8)), "zero": np.full((3, 8), 1.2),
              "esm2": np.full((3, 8), 0.8), "lingshu": np.full((3, 8), 0.75),
              "fusion": np.full((3, 8), 0.5)}
    for seed in screen.SHUFFLES:
        errors[f"esm2_shuffled_{seed}"] = np.ones((3, 8))
        errors[f"lingshu_shuffled_{seed}"] = np.ones((3, 8))
        errors[f"fusion_shuffled_lingshu_{seed}"] = np.full((3, 8), 0.8)
    return errors


def test_fold_training_blocks_have_equal_energy_and_zero_mean():
    esm2, language, effects, _ = synthetic()
    basis = screen.BlockBasis((esm2, language * 1000), effects)
    transformed = basis.transform((esm2, language * 1000))
    np.testing.assert_allclose(transformed.mean(axis=0), 0, atol=1e-15)
    np.testing.assert_allclose(np.mean(np.sum(transformed[:, :5] ** 2, axis=1)), 0.5)
    np.testing.assert_allclose(np.mean(np.sum(transformed[:, 5:] ** 2, axis=1)), 0.5)
    np.testing.assert_allclose(np.mean(np.sum(transformed ** 2, axis=1)), 1.0)
    single = screen.BlockBasis((esm2,), effects)
    np.testing.assert_allclose(np.mean(np.sum(single.transform((esm2,)) ** 2, axis=1)), 1.0)


def test_query_batches_do_not_refit_centers_scales_or_change_predictions():
    esm2, language, effects, _ = synthetic()
    basis = screen.BlockBasis((esm2[:12], language[:12]), effects[:12])
    fitted = basis.fit(AdapterSpec(3, 2, 0.5, 1.0, 0.25))
    centers = [value.copy() for value in basis.centers]
    scales = tuple(basis.scales)
    query = (esm2[12:13], language[12:13])
    alone = fitted.predict_effect(query)
    batched = fitted.predict_effect((np.concatenate([esm2[12:], np.ones((1, 5)) * 1e6]),
                                     np.concatenate([language[12:], np.ones((1, 9)) * -1e6])))
    np.testing.assert_allclose(alone[0], batched[0], rtol=1e-12, atol=1e-12)
    for before, after in zip(centers, basis.centers, strict=True):
        np.testing.assert_array_equal(before, after)
    assert basis.scales == scales
    np.testing.assert_allclose(basis.centers[0], esm2[:12].mean(axis=0))
    np.testing.assert_allclose(basis.centers[1], language[:12].mean(axis=0))


def test_duplicate_dimensions_do_not_favor_a_feature_block():
    esm2, language, effects, _ = synthetic()
    spec = AdapterSpec(4, 3, 1.0, 1.0, 0.25)
    reference = screen.BlockBasis((esm2, language), effects).fit(spec)
    duplicate = screen.BlockBasis((np.tile(esm2, (1, 7)), language), effects).fit(spec)
    np.testing.assert_allclose(reference.predict_effect((esm2, language)),
        duplicate.predict_effect((np.tile(esm2, (1, 7)), language)), rtol=1e-10, atol=1e-10)
    scaled = screen.BlockBasis((esm2 * 100, language * 0.02), effects).fit(spec)
    np.testing.assert_allclose(reference.predict_effect((esm2, language)),
        scaled.predict_effect((esm2 * 100, language * 0.02)), rtol=1e-10, atol=1e-10)


def test_all_families_use_same_total_rank_not_rank_per_block():
    esm2, language, effects, names = synthetic()
    arms, _ = screen.make_arms(esm2, language, names)
    spec = AdapterSpec(3, 2, 1.0, 1.0, 0.25)
    for family in screen.FAMILIES:
        fitted = screen.BlockBasis(arms[family], effects).fit(spec)
        assert fitted.core.feature_basis.shape[1] == 3
        assert fitted.core.effect_basis.shape[1] == 2
        assert fitted.core.coefficients.shape == (3, 2)
    assert len(list(screen.candidates(1.0))) == 25
    assert sum(spec.residual_gain == 0 for spec in screen.candidates(1.0)) == 1


@pytest.mark.parametrize("mean_gain", [0.0, 0.5, 1.0])
def test_constant_fallback_is_exact_for_each_family(mean_gain):
    esm2, language, effects, names = synthetic()
    arms, _ = screen.make_arms(esm2, language, names)
    spec = AdapterSpec(3, 2, 10.0, mean_gain, 0.0)
    expected = np.broadcast_to(mean_gain * effects.mean(axis=0), effects.shape)
    for family in screen.FAMILIES:
        fitted = screen.BlockBasis(arms[family], effects).fit(spec)
        prediction = fitted.predict_effect(tuple(block * 123 for block in arms[family]))
        np.testing.assert_array_equal(prediction, expected)
        np.testing.assert_array_equal(fitted.core.coefficients, np.zeros_like(fitted.core.coefficients))


@pytest.mark.parametrize("bad", [(), (np.zeros((3, 2)), np.zeros((4, 2))),
    (np.asarray([[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]),)])
def test_invalid_training_feature_blocks_rejected(bad):
    with pytest.raises(ValueError):
        screen.BlockBasis(bad, np.ones((3, 2)))


def test_transform_rejects_wrong_block_count_or_width():
    esm2, language, effects, _ = synthetic()
    basis = screen.BlockBasis((esm2, language), effects)
    with pytest.raises(ValueError):
        basis.transform((esm2,))
    with pytest.raises(ValueError):
        basis.transform((esm2[:, :3], language))


def test_negative_controls_use_same_permutation_and_keep_fusion_esm2_genuine():
    esm2, language, _, names = synthetic()
    arms, associations = screen.make_arms(esm2, language, names)
    assert len(arms) == 12
    assert set(associations) == {str(seed) for seed in screen.SHUFFLES}
    for seed in screen.SHUFFLES:
        permutation = np.random.default_rng(seed).permutation(len(names))
        np.testing.assert_array_equal(arms[f"esm2_shuffled_{seed}"][0], esm2[permutation])
        np.testing.assert_array_equal(arms[f"lingshu_shuffled_{seed}"][0], language[permutation])
        np.testing.assert_array_equal(arms[f"fusion_shuffled_lingshu_{seed}"][0], esm2)
        np.testing.assert_array_equal(arms[f"fusion_shuffled_lingshu_{seed}"][1], language[permutation])
        assert associations[str(seed)] == dict(zip(names.tolist(), names[permutation].tolist(), strict=True))


def test_arms_reject_duplicate_or_unaligned_target_identities():
    esm2, language, _, names = synthetic()
    with pytest.raises(ValueError):
        screen.make_arms(esm2, language[:-1], names)
    with pytest.raises(ValueError):
        screen.make_arms(esm2, language, names[:-1])
    names[1] = names[0]
    with pytest.raises(ValueError):
        screen.make_arms(esm2, language, names)


def test_promotion_is_research_only_even_when_every_screen_passes():
    decisions = screen.promotion_decisions(valid_errors())
    assert set(decisions) == set(screen.FAMILIES)
    for family in screen.FAMILIES:
        assert decisions[family]["conditioning_screen_passed"] is True
        assert decisions[family]["submission_allowed"] is False
    assert "esm2" in decisions["fusion"]["comparisons"]
    assert all(f"fusion_shuffled_lingshu_{seed}" in decisions["fusion"]["comparisons"]
               for seed in screen.SHUFFLES)


@pytest.mark.parametrize("control", ["esm2", *(f"fusion_shuffled_lingshu_{s}" for s in screen.SHUFFLES)])
def test_fusion_must_beat_esm2_and_each_incremental_lingshu_control(control):
    errors = valid_errors()
    errors[control] = np.full((3, 8), 0.4)
    assert screen.promotion_decisions(errors)["fusion"]["conditioning_screen_passed"] is False


def test_every_repeat_and_one_percent_margin_are_required():
    errors = valid_errors()
    errors["esm2"][0] = 1.01
    assert screen.promotion_decisions(errors)["esm2"]["conditioning_screen_passed"] is False
    errors = valid_errors()
    errors["lingshu"][:] = 0.995
    decision = screen.promotion_decisions(errors)["lingshu"]
    assert decision["checks"]["at_least_one_percent_vs_constant"] is False
    assert decision["conditioning_screen_passed"] is False


@pytest.mark.parametrize("change", ["missing", "extra", "nan", "inf", "negative", "shape", "empty", "vector"])
def test_decisions_reject_incomplete_or_invalid_error_panels(change):
    errors = valid_errors()
    if change == "missing":
        del errors[f"fusion_shuffled_lingshu_{screen.SHUFFLES[0]}"]
    elif change == "extra":
        errors["unregistered"] = np.ones((3, 8))
    elif change in ("nan", "inf", "negative"):
        errors["fusion"][0, 0] = {"nan": np.nan, "inf": np.inf, "negative": -0.1}[change]
    elif change == "shape":
        errors["fusion"] = np.ones((2, 8))
    elif change == "empty":
        errors = {key: np.empty((0, 8)) for key in errors}
    else:
        errors = {key: np.ones(8) for key in errors}
    with pytest.raises(ValueError):
        screen.promotion_decisions(errors)


def test_inner_selection_fits_every_pca_only_on_inner_training_rows(monkeypatch):
    esm2, language, effects, _ = synthetic(12)
    splits = screen.folds(12, 3, 83)
    observed = []
    original_pca = adapter_module.pca_basis
    monkeypatch.setattr(screen, "candidates", small_grid)

    def pca_spy(centered, max_rank):
        observed.append(centered.copy())
        return original_pca(centered, max_rank)

    monkeypatch.setattr(adapter_module, "pca_basis", pca_spy)
    selected, scores = screen.select((esm2, language), effects, splits, 0.5)
    assert selected in list(small_grid(0.5))
    assert len(scores) == 2
    assert len(observed) == 2 * len(splits)
    for index, heldout in enumerate(splits):
        train = np.setdiff1d(np.arange(12), heldout)
        normalized = []
        for block in (esm2[train], language[train]):
            centered = block - block.mean(axis=0)
            scale = np.sqrt(np.mean(np.sum(centered ** 2, axis=1)))
            normalized.append(centered / scale)
        features = np.concatenate(normalized, axis=1) / np.sqrt(2)
        features -= features.mean(axis=0)
        features /= np.sqrt(np.mean(np.sum(features ** 2, axis=1)))
        np.testing.assert_allclose(observed[2 * index], features, atol=1e-14)
        np.testing.assert_allclose(observed[2 * index + 1], effects[train] - effects[train].mean(axis=0))
        assert observed[2 * index].shape[0] == len(train)
        assert observed[2 * index + 1].shape[0] == len(train)


def test_small_nested_cv_is_deterministic_with_shared_partitions_and_budget(monkeypatch):
    esm2, language, effects, names = synthetic(18)
    monkeypatch.setattr(screen, "candidates", small_grid)
    arms, _ = screen.make_arms(esm2, language, names)
    result, errors = screen.evaluate(arms, effects, names, seeds=(991, 992), outer_folds=3)
    repeated, repeated_errors = screen.evaluate(arms, effects, names, seeds=(991, 992), outer_folds=3)
    assert result == repeated
    assert len(result["folds"]) == 12 * 2 * 3
    for arm in errors:
        np.testing.assert_array_equal(errors[arm], repeated_errors[arm])
        assert errors[arm].shape == (2, 18)
        assert np.isfinite(errors[arm]).all()
    for seed in (991, 992):
        by_fold = {fold: [row for row in result["folds"] if row["repeat_seed"] == seed
                          and row["outer_fold"] == fold] for fold in range(3)}
        seen = []
        for rows in by_fold.values():
            assert {row["arm"] for row in rows} == set(arms)
            reference = rows[0]
            seen.extend(reference["heldout_targets"])
            assert set(reference["heldout_targets"]).isdisjoint(reference["train_targets"])
            assert sorted(sum(reference["inner_fold_targets"], [])) == sorted(reference["train_targets"])
            for row in rows:
                assert row["train_targets"] == reference["train_targets"]
                assert row["heldout_targets"] == reference["heldout_targets"]
                assert row["inner_fold_targets"] == reference["inner_fold_targets"]
                assert len(row["candidate_inner_scores"]) == 2
                assert [entry["spec"] for entry in row["candidate_inner_scores"]] == [
                    entry["spec"] for entry in reference["candidate_inner_scores"]]
        assert sorted(seen) == sorted(names.tolist())


def test_research_exports_are_pickle_free_exclusive_and_never_approval(tmp_path):
    esm2, language, effects, names = synthetic()
    fitted = screen.BlockBasis((esm2, language), effects).fit(AdapterSpec(3, 2, 1.0, 1.0, 0.25))
    target = tmp_path / "fusion.research-only.npz"
    metadata = {"family": "fusion", "training_targets": names.tolist(), "conditioning_screen_passed": True}
    fitted.save(target, metadata)
    with np.load(target, allow_pickle=False) as saved:
        assert str(saved["schema"]) == "vcc-biological-conditioning-research-v5"
        assert not bool(saved["submission_allowed"])
        assert json.loads(str(saved["metadata"])) == metadata
        assert json.loads(str(saved["spec"])) == asdict(fitted.core.spec)
        np.testing.assert_array_equal(saved["block_widths"], [5, 9])
        for key in saved.files:
            assert saved[key].dtype != object
        offset = 0
        normalized = []
        for block, width, scale in zip((esm2, language), saved["block_widths"], saved["block_scales"], strict=True):
            center = saved["block_centers"][offset:offset + width]
            normalized.append((block - center) / scale)
            offset += width
        x = np.concatenate(normalized, axis=1) / np.sqrt(2)
        latent = ((x - saved["feature_center"]) / saved["feature_scale"]) @ saved["feature_basis"] @ saved["coefficients"]
        prediction = (fitted.core.spec.mean_gain * saved["effect_center"] +
                      fitted.core.spec.residual_gain * latent @ saved["effect_basis"].T)
        np.testing.assert_allclose(prediction, fitted.predict_effect((esm2, language)))
    with pytest.raises(FileExistsError):
        fitted.save(target, metadata)


@pytest.mark.parametrize("missing", [None, "SYNTHETIC_001", "two"])
def test_main_rejects_any_change_to_registered_145_target_taz_exclusion(tmp_path, monkeypatch, missing):
    names = np.asarray([f"SYNTHETIC_{i:03d}" for i in range(145)] + ["TAZ"])
    cache, embeddings, receipt = (tmp_path / name for name in ("cache.npz", "language.npz", "receipt.json"))
    np.savez(cache, train_targets=names)
    np.savez(embeddings, gene_names=names, embeddings=np.ones((146, 3)))
    receipt.write_text("{}")
    feature_names = names.tolist()
    if missing == "two":
        feature_names.remove("TAZ")
        feature_names.remove("SYNTHETIC_001")
    elif missing is not None:
        feature_names.remove(missing)
    monkeypatch.setattr(screen, "load_authenticated_arc_feature_map", lambda path: ({name: None for name in feature_names}, {}))
    hashes = {cache: screen.CACHE_SHA, embeddings: screen.EMBEDDING_SHA, receipt: screen.RECEIPT_SHA}
    monkeypatch.setattr(screen, "digest", lambda path: hashes.get(Path(path), "a" * 64))
    monkeypatch.setattr(screen.socket, "getfqdn", lambda: "cbsuvlaminck3.biohpc.cornell.edu")
    monkeypatch.setattr(screen, "training_effects", lambda archive: pytest.fail("Expression read before panel validation"))
    monkeypatch.setattr(sys, "argv", ["synthetic-screen", "--cache", str(cache), "--esm2-features", str(tmp_path / "esm2.pt"),
        "--lingshu-embeddings", str(embeddings), "--lingshu-receipt", str(receipt), "--output-dir", str(tmp_path / "output")])
    with pytest.raises(ValueError, match="145-target shared panel changed"):
        screen.main()
    assert not (tmp_path / "output" / "contract.json").exists()
