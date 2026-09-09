"""Control-anchored public group-effect flow using the genuine scDFM PAD.

The backbone sees residual expression and a zero control branch. Consequently
adding the same expression offset to x_t and control cannot change velocity.
There is no hard-coded target knockdown or fitted challenge response.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from lingshu_scdfm_model import LingshuScDFM, grouped_rows, integrate, require

CHECKPOINT_SCHEMA = "vcc-lingshu-scdfm-public-group-effect-checkpoint-v2"
METHOD = "control_anchored_public_group_effect_cfm_v2"


def feature_mapping(embedding_map, mode: str, seed: int):
    """Permute target-feature associations, never expression labels or rows."""
    require(mode in ("true", "shuffled", "constant"), "Unknown feature mode")
    names = sorted(embedding_map)
    assigned = names.copy()
    if mode == "shuffled":
        assigned = np.random.default_rng(seed).permutation(names).tolist()
    associations = dict(zip(names, assigned, strict=True))
    encoded = json.dumps(associations, sort_keys=True, separators=(",", ":")).encode()
    return {name: embedding_map[associations[name]] for name in names}, {
        "mode": mode, "seed": seed, "available_gene_count": len(names),
        "gene_feature_mapping": associations,
        "mapping_sha256": hashlib.sha256(encoded).hexdigest(),
        "challenge_expression_or_labels_used": False,
        "constant_mode_input": "exact_zero_after_normalization" if mode == "constant" else None,
    }


def fit_feature_normalization(embedding_map, training_targets):
    """One equally weighted feature per training target; no validation fitting."""
    names = sorted(set(map(str, training_targets)))
    require(bool(names), "No training targets for feature normalization")
    values = np.stack([embedding_map[name] for name in names]).astype(np.float64)
    center = values.mean(axis=0)
    scale = float(np.sqrt(np.mean((values - center) ** 2)))
    # A one-target synthetic test has no dispersion; identity scaling is finite.
    degenerate = scale < 1e-12
    if degenerate:
        scale = 1.0
    return center.astype(np.float32), np.float32(scale), {
        "fit_scope": "vectors_assigned_to_unique_public_training_targets_after_registered_mapping",
        "training_targets": names, "training_target_count": len(names),
        "scale_kind": "scalar_root_mean_square_centered_feature_element",
        "scale": scale, "degenerate_scale_fallback": degenerate,
        "validation_expression_used_for_fit": False, "challenge_expression_used_for_fit": False,
    }


class LingshuScDFMEffect(LingshuScDFM):
    def __init__(self, config: dict, source: Path, feature_center=None, feature_scale=None):
        super().__init__(config, source)
        dimension = config["embedding_dimension"]
        center = torch.zeros(dimension) if feature_center is None else torch.as_tensor(feature_center, dtype=torch.float32)
        scale = torch.tensor(1.0) if feature_scale is None else torch.as_tensor(feature_scale, dtype=torch.float32)
        require(center.shape == (dimension,) and scale.numel() == 1, "Invalid fitted feature transform")
        require(bool(torch.isfinite(center).all()) and bool(torch.isfinite(scale).all()) and float(scale) > 0,
                "Nonfinite fitted feature transform")
        self.register_buffer("feature_center", center.clone())
        self.register_buffer("feature_scale", scale.reshape(()).clone())

    def normalize_features(self, features):
        if self.config["feature_mode"] == "constant":
            return torch.zeros_like(features)
        return (features - self.feature_center) / self.feature_scale

    def forward(self, x_t, control, target_features, time):
        residual = x_t - control
        return super().forward(residual, torch.zeros_like(control),
                               self.normalize_features(target_features), time)


def group_effects(data, split: str, groups=None):
    """All cached cells per group, not arbitrary unpaired cell differences."""
    require(split in ("train", "val"), "Unknown data split")
    if groups is None:
        groups = grouped_rows(data[f"{split}_targets"], data[f"{split}_contexts"],
                              data["val_kind"] if split == "val" else None)
    effects = []
    for _, rows in groups:
        treated_mean = data[f"{split}_x"][rows].astype(np.float64).mean(axis=0)
        control_mean = data[f"{split}_control"][rows].astype(np.float64).mean(axis=0)
        effects.append(treated_mean - control_mean)
    return groups, np.asarray(effects, dtype=np.float32)


@torch.inference_mode()
def predict_group_effects(model, features, *, gene_count, steps=20, bf16=False):
    zero = torch.zeros((len(features), gene_count), device=features.device)
    return integrate(model, zero, features, steps=steps, noise_std=0.0,
                     generator=torch.Generator(device=features.device).manual_seed(0), bf16=bf16)
