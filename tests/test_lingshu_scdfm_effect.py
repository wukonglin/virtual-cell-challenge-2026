"""Focused tests for the isolated control-anchored public group-effect route."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch
from torch import nn

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from lingshu_scdfm_effect_model import (
    CHECKPOINT_SCHEMA, LingshuScDFMEffect, feature_mapping, fit_feature_normalization,
    group_effects, predict_group_effects,
)
from train_lingshu_scdfm_effect import parser, train, training_times, validate_group_rollouts


class ToyPAD(nn.Module):
    """Test-only substitute; the production path imports authenticated PAD."""
    def __init__(self, ntoken, d_model, **kwargs):
        super().__init__()
        self.perturbation_embedder = nn.Module()
        self.perturbation_embedder.embedding = nn.Embedding(ntoken, d_model)
        self.p_mask_embed = nn.Parameter(torch.zeros(d_model))
        self.p_head = nn.Linear(d_model, 1)
        self.effect = nn.Linear(d_model, ntoken)

    def forward(self, ids, x, time, control, perturbation_emb):
        self.last_inputs = (x.detach().clone(), control.detach().clone())
        return self.effect(perturbation_emb) + 0.1 * x


def config(mode="true"):
    return {"gene_count": 2, "embedding_dimension": 3, "hidden_size": 16,
            "heads": 2, "layers": 1, "dropout": 0.0, "feature_mode": mode}


def synthetic_cache():
    return {"gene_names": np.array(["G1", "G2"]), "gene_indices": np.array([0, 1]),
            "normalization_gene_names": np.array(["G1", "G2"]),
            "train_x": np.array([[1, 2], [2, 1], [2, 2], [3, 1]], dtype=np.float32),
            "train_control": np.ones((4, 2), dtype=np.float32),
            "train_targets": np.array(["T1", "T1", "T2", "T2"]),
            "train_contexts": np.array(["K562"] * 4),
            "val_x": np.array([[1, 2], [2, 1], [2, 2], [3, 1]], dtype=np.float32),
            "val_control": np.ones((4, 2), dtype=np.float32),
            "val_targets": np.array(["T1", "T1", "T3", "T3"]),
            "val_contexts": np.array(["RPE1", "RPE1", "K562", "K562"]),
            "val_kind": np.array(["context", "context", "target", "target"]),
            "metadata_json": np.array(json.dumps({"challenge_treated_used": False,
                "normalization": {"space": "log1p_library_normalized_full_axis", "target_sum": 10000,
                                  "full_axis_scope": "shared_public_challenge_gene_axis"}}))}


class EffectContractTests(unittest.TestCase):
    def test_feature_transform_fits_unique_training_targets_only(self):
        features = {"A": np.array([1, 2]), "B": np.array([3, 4]), "heldout": np.array([500, 900])}
        center, scale, metadata = fit_feature_normalization(features, ["A", "A", "B"])
        np.testing.assert_array_equal(center, [2, 3])
        self.assertEqual(float(scale), 1.0)
        self.assertEqual(metadata["training_targets"], ["A", "B"])
        features["heldout"] *= 100
        again = fit_feature_normalization(features, ["A", "B"])
        np.testing.assert_array_equal(center, again[0])
        self.assertEqual(float(scale), float(again[1]))

    def test_shuffling_is_seeded_bijection_without_changing_source_features(self):
        features = {str(i): np.array([i, i + 1], dtype=np.float32) for i in range(10)}
        result, metadata = feature_mapping(features, "shuffled", 7)
        again, repeated = feature_mapping(features, "shuffled", 7)
        self.assertEqual(metadata, repeated)
        self.assertEqual(set(metadata["gene_feature_mapping"].values()), set(features))
        self.assertTrue(any(not np.array_equal(result[k], features[k]) for k in features))
        for key in result:
            np.testing.assert_array_equal(result[key], again[key])
        np.testing.assert_array_equal(features["0"], [0, 1])

    def test_group_effect_uses_all_training_rows_and_not_validation(self):
        data = synthetic_cache()
        groups, effects = group_effects(data, "train")
        np.testing.assert_array_equal(effects, [[0.5, 0.5], [1.5, 0.5]])
        data["val_x"][:] = 8000
        np.testing.assert_array_equal(group_effects(data, "train")[1], effects)
        np.testing.assert_array_equal(groups[0][1], [0, 1])

    def test_exact_half_zero_start_supervision_and_even_batch_required(self):
        times = training_times(32, torch.device("cpu"))
        self.assertTrue(torch.equal(times[:16], torch.zeros(16)))
        self.assertTrue(bool(((times[16:] >= 0) & (times[16:] < 1)).all()))
        with self.assertRaisesRegex(ValueError, "even"):
            training_times(3, torch.device("cpu"))

    def test_translation_equivariance_constant_mode_and_persistent_transform(self):
        with mock.patch("lingshu_scdfm_model._official_model_class", return_value=ToyPAD):
            model = LingshuScDFMEffect(config(), Path("unused"), [1, 2, 3], 2.0).eval()
            x, control, features, t = torch.randn(4, 2), torch.randn(4, 2), torch.randn(4, 3), torch.rand(4)
            first = model(x, control, features, t)
            x_seen, c_seen = model.pad.last_inputs
            torch.testing.assert_close(x_seen, x - control)
            self.assertTrue(torch.equal(c_seen, torch.zeros_like(control)))
            shift = torch.randn(4, 2)
            torch.testing.assert_close(first, model(x + shift, control + shift, features, t), atol=1e-6, rtol=1e-6)
            self.assertIn("feature_center", model.state_dict())
            self.assertIn("feature_scale", model.state_dict())
            constant = LingshuScDFMEffect(config("constant"), Path("unused"), [1, 2, 3], 2.0).eval()
            self.assertTrue(torch.equal(constant.normalize_features(features), torch.zeros_like(features)))
            torch.testing.assert_close(constant(x, control, features, t), constant(x, control, features * 13, t))

    def test_selection_uses_actual_zero_start_flow_against_group_means(self):
        class ConstantEffect(nn.Module):
            def forward(self, x, control, features, time):
                return features[:, :2]
        data = synthetic_cache()
        groups, effects = group_effects(data, "val")
        feature_map = {key[-1]: effects[index] for index, (key, _) in enumerate(groups)}
        metrics = validate_group_rollouts(ConstantEffect(), groups, effects, feature_map,
                                          device=torch.device("cpu"), batch_size=2, bf16=False)
        self.assertLess(metrics["selection_group_effect_mse"], 1e-11)
        self.assertTrue(metrics["checkpoint_selection_affected"])
        self.assertFalse(metrics["teacher_forced_interpolants_used"])
        self.assertEqual([row["all_cached_cells_in_group"] for row in metrics["groups"]], [2, 2])

    def test_training_checkpoint_and_diagnostic_contract_with_test_backbone(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            np.savez(root / "cache.npz", **synthetic_cache())
            np.savez(root / "embeddings.npz", gene_names=np.array(["T1", "T2", "T3"]),
                     embeddings=np.eye(3, dtype=np.float32))
            args = parser().parse_args(["--cache", str(root / "cache.npz"), "--embeddings", str(root / "embeddings.npz"),
                "--source", str(root / "source"), "--output-dir", str(root / "fit"), "--steps", "2",
                "--validate-every", "1", "--batch-size", "2", "--validation-batch-size", "2", "--hidden-size", "16",
                "--heads", "2", "--layers", "1", "--cpu", "--debug-test-features"])
            with mock.patch("lingshu_scdfm_model._official_model_class", return_value=ToyPAD), \
                 mock.patch("train_lingshu_scdfm_effect.authenticate_upstream", return_value={"test_only": True}), \
                 redirect_stdout(io.StringIO()):
                summary = train(args)
                with self.assertRaisesRegex(ValueError, "independent new"):
                    train(args)
            checkpoint = torch.load(root / "fit/best.pt", weights_only=True)
            self.assertEqual(checkpoint["schema"], CHECKPOINT_SCHEMA)
            self.assertEqual(checkpoint["training_config"]["t0_fraction"], 0.5)
            self.assertEqual(checkpoint["training_config"]["noise_std"], 0)
            self.assertEqual(summary["schema"], "vcc-lingshu-scdfm-training-summary-v1")
            diagnostic = summary["deployment_rollout_diagnostics"]
            self.assertFalse(diagnostic["checkpoint_selection_affected"])
            self.assertTrue(diagnostic["selection_used_same_public_development_groups"])
            self.assertFalse(diagnostic["independent_test"])
            self.assertFalse(summary["challenge_treated_used"])
            self.assertEqual(set(diagnostic["kind_metrics"]), {"context", "target"})
            self.assertEqual(len(checkpoint["code_sha256"]), 4)


@unittest.skipUnless(importlib.util.find_spec("timm") and importlib.util.find_spec("torchvision"),
                     "Official PAD dependencies unavailable")
class OfficialEffectIntegration(unittest.TestCase):
    def test_authentic_pad_gradient_and_translation_equivariance(self):
        source = SCRIPTS.parent / "external/scdfm_lingshu_2cf6bca1"
        if not source.exists():
            self.skipTest("Pinned source unavailable")
        torch.set_num_threads(1)
        model = LingshuScDFMEffect(config(), source, [0.1, 0.2, 0.3], 0.5).eval()
        x, control, features, t = torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 3), torch.rand(2)
        result = model(x, control, features, t)
        torch.testing.assert_close(result, model(x + 2, control + 2, features, t), atol=1e-5, rtol=1e-5)
        result.square().mean().backward()
        gradient = model.target_projection[0].weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(type(model.pad).__module__.endswith("models.origin.model"))


if __name__ == "__main__":
    unittest.main()
