"""Compact, public-data-trained Lingshu conditioning for the official scDFM PAD.

This is a new model trained from scratch, not a transferred Norman checkpoint.
The PAD implementation is imported unmodified from a pinned official checkout.
We train a projection from frozen Lingshu features and disable the optional
gene-correlation graph. Expression is log1p(CP10k), with full-axis library sizes.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch
from torch import nn

UPSTREAM_COMMIT = "2cf6bca1f044e74c4e1dc586892c0495880cf125"
UPSTREAM_TREE = "a04130b07020505a609158cbe31e9e65083c0d79"
CHECKPOINT_SCHEMA = "vcc-lingshu-scdfm-public-adapter-checkpoint-v1"
NORMALIZATION = "log1p_library_normalized_full_axis"
LINGSHU_MODEL_ID = "lingshu-medical-mllm/Lingshu-7B"
LINGSHU_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
LINGSHU_DIMENSION = 3584


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def authenticate_upstream(source: Path) -> dict[str, str]:
    source = source.resolve()
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(source), *args], text=True).strip()
    require(git("rev-parse", "HEAD") == UPSTREAM_COMMIT, "Wrong official scDFM commit")
    require(git("rev-parse", "HEAD^{tree}") == UPSTREAM_TREE, "Wrong official scDFM tree")
    require(not git("status", "--porcelain", "--untracked-files=normal"), "Official scDFM checkout is modified")
    return {"commit": UPSTREAM_COMMIT, "tree": UPSTREAM_TREE}


def _official_model_class(source: Path):
    authenticate_upstream(source)
    source = source.resolve()
    # A different project may also have a top-level namespace called src.
    for name, module in tuple(sys.modules.items()):
        if name == "src" or name.startswith("src."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                require(Path(origin).resolve().is_relative_to(source), "Conflicting src module loaded")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(source))
    module = importlib.import_module("src.models.origin.model")
    require(Path(module.__file__).resolve() == source / "src/models/origin/model.py", "Wrong PAD module imported")
    return module.model


class LingshuScDFM(nn.Module):
    def __init__(self, config: dict[str, Any], source: Path):
        super().__init__()
        require(config["hidden_size"] % config["heads"] == 0, "Hidden size must divide heads")
        require(config["gene_count"] > 1 and config["embedding_dimension"] > 1, "Invalid model dimensions")
        self.config = dict(config)
        hidden = config["hidden_size"]
        self.target_projection = nn.Sequential(
            nn.Linear(config["embedding_dimension"], hidden),
            nn.SiLU(), nn.Linear(hidden, hidden), nn.LayerNorm(hidden),
        )
        self.pad = _official_model_class(source)(
            ntoken=config["gene_count"], d_model=hidden,
            nhead=config["heads"], d_hid=hidden * 4,
            nlayers=config["layers"], dropout=config["dropout"],
            fusion_method="differential_perceiver", perturbation_function="crisper",
            use_perturbation_interaction=False, mask_path=None,
        )
        # The official external-embedding path uses only this table's LayerNorm.
        self.pad.perturbation_embedder.embedding.weight.requires_grad_(False)
        # predict_p is not trained in this public-response experiment.
        self.pad.p_mask_embed.requires_grad_(False)
        for parameter in self.pad.p_head.parameters():
            parameter.requires_grad_(False)
        self.register_buffer("gene_ids", torch.arange(config["gene_count"]), persistent=False)

    def forward(self, x_t, control, target_features, time):
        ids = self.gene_ids[None, :].expand(x_t.shape[0], -1)
        condition = self.target_projection(target_features)
        return self.pad(ids, x_t, time, control, perturbation_emb=condition)


def load_embeddings(path: Path) -> tuple[dict[str, np.ndarray], str]:
    with np.load(path, allow_pickle=False) as archive:
        names = archive["gene_names"].astype(str)
        values = archive["embeddings"].astype(np.float32)
    require(names.ndim == 1 and len(set(names.tolist())) == len(names), "Duplicate or invalid embedding names")
    require(values.ndim == 2 and values.shape[0] == len(names), "Embedding shape mismatch")
    require(np.isfinite(values).all(), "Nonfinite Lingshu embeddings")
    norms = np.linalg.norm(values, axis=1)
    require((norms > 0).all(), "Zero Lingshu embedding")
    require(len({hashlib.sha256(row.tobytes()).digest() for row in values}) == len(names), "Distinct targets share an identical embedding")
    # A frozen, explicitly documented normalization; no fitted transform.
    values /= norms[:, None]
    return dict(zip(names.tolist(), values, strict=True)), sha256_file(path)


def authenticate_embedding_receipt(receipt_path: Path, npz_path: Path,
                                   embedding_map: dict[str, np.ndarray], digest: str) -> dict:
    receipt = json.loads(receipt_path.read_text())
    for key, expected in (("schema", "vcc-lingshu-frozen-gene-embeddings-v1"), ("status", "complete"),
                          ("model_id", LINGSHU_MODEL_ID), ("model_revision", LINGSHU_REVISION),
                          ("embedding_dimension", LINGSHU_DIMENSION),
                          ("role", "frozen_symbol_prompt_language_features"),
                          ("pooling", "last_hidden_state_attention_masked_mean_float32_then_l2"),
                          ("challenge_treated_data_used", False), ("generated_annotations_used", False)):
        require(receipt.get(key) == expected, f"Lingshu receipt has wrong {key}")
    require(receipt.get("gene_names") == list(embedding_map), "Embedding receipt ordered gene names differ")
    require(receipt.get("gene_count") == len(embedding_map), "Embedding receipt gene count differs")
    require(all(row.shape == (LINGSHU_DIMENSION,) for row in embedding_map.values()), "Lingshu dimension is not 3584")
    npz = receipt.get("npz", {})
    require(npz.get("sha256") == digest and npz.get("size_bytes") == npz_path.stat().st_size,
            "Embedding receipt does not bind NPZ bytes")
    require(npz.get("keys") == ["gene_names", "embeddings"], "Unexpected embedding NPZ fields")
    return {"sha256": sha256_file(receipt_path), "model_id": LINGSHU_MODEL_ID,
            "model_revision": LINGSHU_REVISION, "embedding_dimension": LINGSHU_DIMENSION,
            "npz_sha256": digest, "pooling": receipt["pooling"]}


def load_cache(path: Path) -> tuple[dict[str, Any], str]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    require("metadata_json" in data, "Training cache lacks provenance")
    metadata = json.loads(str(data["metadata_json"].item()))
    require(metadata.get("challenge_treated_used") is False, "Cache is not public-data-only")
    norm = metadata.get("normalization", {})
    require(norm.get("space") == NORMALIZATION and norm.get("target_sum") == 10000, "Unsupported cache normalization")
    require(norm.get("full_axis_scope") == "shared_public_challenge_gene_axis", "Training/inference must share the normalization axis")
    genes = data["gene_names"].astype(str)
    require("normalization_gene_names" in data, "Missing shared normalization gene axis")
    normalization_genes = data["normalization_gene_names"].astype(str)
    require(normalization_genes.ndim == 1 and len(set(normalization_genes.tolist())) == len(normalization_genes),
            "Invalid normalization gene axis")
    require(set(genes.tolist()) <= set(normalization_genes.tolist()), "Modeled genes lie outside normalization axis")
    indices = data["gene_indices"]
    require(genes.ndim == 1 and len(set(genes.tolist())) == len(genes), "Invalid selected gene names")
    require(indices.shape == genes.shape and indices.dtype.kind in "iu", "Invalid selected gene indices")
    require((indices >= 0).all() and len(set(indices.tolist())) == len(genes), "Duplicate gene indices")
    require(np.all(indices[1:] > indices[:-1]), "Selected genes must retain full-axis order")
    for split in ("train", "val"):
        x, control = data[f"{split}_x"], data[f"{split}_control"]
        require(x.ndim == 2 and x.shape[1] == len(genes) and x.shape == control.shape and len(x) > 0, "Cache expression dimensions disagree")
        for value in (x, control):
            require(np.isfinite(value).all() and (value >= 0).all(), "Cache is not finite nonnegative log-expression")
            require(float(value.max()) <= np.log1p(10000) + 0.01, "Cache values exceed CP10k log1p range")
        for suffix in ("targets", "contexts"):
            labels = data[f"{split}_{suffix}"].astype(str)
            require(labels.shape == (len(x),) and all(labels), "Missing row labels")
            data[f"{split}_{suffix}"] = labels
        data[f"{split}_x"] = x.astype(np.float32)
        data[f"{split}_control"] = control.astype(np.float32)
    kinds = data["val_kind"].astype(str)
    require(kinds.shape == (len(data["val_x"]),) and set(kinds.tolist()) == {"context", "target"}, "Both whole-context and whole-target validation are required")
    for kind, label in (("context", "contexts"), ("target", "targets")):
        selected = data[f"val_{label}"][kinds == kind]
        require(not set(selected.tolist()) & set(data[f"train_{label}"].tolist()), f"Whole-{kind} validation leaks into training")
    data["val_kind"] = kinds
    data["gene_names"] = genes
    data["normalization_gene_names"] = normalization_genes
    data["metadata"] = metadata
    return data, sha256_file(path)


def grouped_rows(targets: np.ndarray, contexts: np.ndarray, kinds: np.ndarray | None = None):
    groups: dict[tuple[str, ...], list[int]] = {}
    for index, (target, context) in enumerate(zip(targets, contexts, strict=True)):
        key = (str(context), str(target)) if kinds is None else (str(kinds[index]), str(context), str(target))
        groups.setdefault(key, []).append(index)
    return [(key, np.asarray(rows, dtype=np.int64)) for key, rows in sorted(groups.items())]


def mmd2_unbiased(generated: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Official scDFM objective: four dynamic-bandwidth unbiased RBF kernels."""
    generated, target = generated.float(), target.float()
    require(len(generated) > 1 and len(target) > 1, "MMD requires at least two cells")
    dxx = torch.cdist(generated, generated).square()
    dyy = torch.cdist(target, target).square()
    dxy = torch.cdist(generated, target).square()
    mask = ~torch.eye(len(target), dtype=torch.bool, device=target.device)
    median = dyy.detach()[mask].median().clamp_min(1e-6)
    estimates = []
    for scale in (0.5, 1.0, 2.0, 4.0):
        denominator = 2 * median * scale
        xx, yy, xy = torch.exp(-dxx / denominator), torch.exp(-dyy / denominator), torch.exp(-dxy / denominator)
        estimates.append((xx.sum() - xx.diag().sum()) / (len(xx) * (len(xx) - 1))
                         + (yy.sum() - yy.diag().sum()) / (len(yy) * (len(yy) - 1)) - 2 * xy.mean())
    return torch.stack(estimates).mean()


@torch.inference_mode()
def integrate(model, control, target_features, *, steps: int, noise_std: float,
              generator: torch.Generator, bf16: bool = False):
    require(steps > 0, "ODE steps must be positive")
    x = control + torch.randn(control.shape, generator=generator, device=control.device) * noise_std
    for step in range(steps):
        time = torch.full((len(x),), step / steps, device=x.device)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity = model(x, control, target_features, time)
        x = x + velocity.float() / steps
        require(bool(torch.isfinite(x).all()), "Nonfinite Euler trajectory")
    return x
