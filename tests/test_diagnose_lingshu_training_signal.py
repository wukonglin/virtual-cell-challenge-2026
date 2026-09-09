import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnose_lingshu_training_signal import TRAIN_KEYS, fit_predict, folds, normalize, training_effects


def test_loader_never_requests_validation_arrays():
    class Archive:
        def __init__(self):
            self.read = []

        def __getitem__(self, key):
            self.read.append(key)
            assert key in TRAIN_KEYS
            if key == "train_targets":
                return np.asarray([f"g{i}" for i in range(15)])
            if key == "train_contexts":
                return np.asarray(["K562"] * 15)
            return np.ones((15, 3)) if key == "train_x" else np.zeros((15, 3))

    archive = Archive()
    names, effects, counts = training_effects(archive)
    assert set(archive.read) == set(TRAIN_KEYS)
    assert effects.shape == (15, 3)
    np.testing.assert_equal(effects, 1)
    assert sum(counts) == len(names) == 15


def test_query_distribution_does_not_change_training_transform():
    train = np.arange(24).reshape(6, 4).astype(float)
    first, _ = normalize(train, np.zeros((2, 4)))
    second, _ = normalize(train, np.full((2, 4), 1e8))
    np.testing.assert_array_equal(first, second)


@pytest.mark.parametrize("kind", ["linear", "rbf"])
def test_constant_features_return_training_mean(kind):
    prediction, fitted = fit_predict(np.ones((12, 6)), np.arange(36).reshape(12, 3),
                                     np.ones((4, 6)), kind, 0.01)
    expected = np.arange(36).reshape(12, 3).mean(axis=0)
    np.testing.assert_allclose(prediction, np.broadcast_to(expected, prediction.shape))
    np.testing.assert_allclose(fitted, np.broadcast_to(expected, fitted.shape))


def test_folds_partition_every_target_once():
    parts = folds(146, 5, 20260909)
    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(146))
    assert max(map(len, parts)) - min(map(len, parts)) <= 1


def test_linear_recovers_synthetic_signal():
    rng = np.random.default_rng(2)
    features = rng.normal(size=(60, 4))
    weights = rng.normal(size=(4, 3))
    effects = features @ weights + 0.7
    prediction, _ = fit_predict(features[:50], effects[:50], features[50:], "linear", 1e-6)
    np.testing.assert_allclose(prediction, effects[50:], atol=1e-3)
