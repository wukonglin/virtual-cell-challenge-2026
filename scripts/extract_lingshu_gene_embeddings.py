#!/usr/bin/env python3
"""Extract frozen Lingshu language features; no perturbation-response training.

Only the local, byte-authenticated official Lingshu-7B snapshot is accepted.
The JSON receipt is the completion marker for its matching NPZ. Both outputs
are published atomically without overwriting existing files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import io
import json
import os
import re
import socket
import tempfile
import zipfile
from pathlib import Path

import numpy as np

MODEL_ID = "lingshu-medical-mllm/Lingshu-7B"
MODEL_REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
EMBEDDING_DIMENSION = 3584
TRANSFORMERS_VERSION = "4.52.1"
MAX_TOKENS = 128
PROMPT_TEMPLATE = (
    "Human gene symbol: {gene}. Perturbation: CRISPR interference (CRISPRi), "
    "reducing expression of this gene in a human cell."
)
SCHEMA = "vcc-lingshu-frozen-gene-embeddings-v1"

# Immutable upstream identities from the official Hugging Face model API.
# Small Git objects use Git's blob SHA-1; LFS payloads use published SHA-256.
MODEL_FILES = {
    "README.md": (26900, "git_blob_sha1", "a9a6576517b126bcca77987ef9a5579affe25d9f"),
    "added_tokens.json": (605, "git_blob_sha1", "482ced4679301bf287ebb310bdd1790eb4514232"),
    "chat_template.json": (1049, "git_blob_sha1", "13303be6f396b038cf397b34a8096819d403836b"),
    "config.json": (1491, "git_blob_sha1", "3518ba0a016f9080409b1ef445d34f72a9f39d59"),
    "generation_config.json": (244, "git_blob_sha1", "54782b57c02403e0a78f4ec674857e159b4fde80"),
    "merges.txt": (1671853, "git_blob_sha1", "31349551d90c7606f325fe0f11bbb8bd5fa0d7c7"),
    "model-00001-of-00004.safetensors": (4968243304, "sha256", "a795e5ca8f2c81eb80ade52c0a48b6711de3a646594cd3445a780b9d75b5c42f"),
    "model-00002-of-00004.safetensors": (4991495816, "sha256", "89da916df65b1726598048622bf45aade61e78b6167ada2c11a9cca3af571f37"),
    "model-00003-of-00004.safetensors": (4932751040, "sha256", "b1cd9fc701c9b4f5f55a28f5016dd524dccfb05b9412b0b6a3fcda58619144cf"),
    "model-00004-of-00004.safetensors": (1691924384, "sha256", "8cc93a95847cae0c09fb455226ff4f66caba67b26054ce694c2264e61d476a08"),
    "model.safetensors.index.json": (57619, "git_blob_sha1", "6a84000f74f7809388c808edf0c022f71e09b5ff"),
    "preprocessor_config.json": (575, "git_blob_sha1", "ee08cdd031dd2f138ce236b0c995b34b4b34a6a4"),
    "special_tokens_map.json": (613, "git_blob_sha1", "ac23c0aaa2434523c494330aeb79c58395378103"),
    "tokenizer.json": (11421896, "sha256", "9c5ae00e602b8860cbd784ba82a8aa14e8feecec692e7076590d014d7b7fdafa"),
    "tokenizer_config.json": (5803, "git_blob_sha1", "5c428cd55e4353de9257cadfe570a6260045bbe8"),
    "vocab.json": (2776833, "git_blob_sha1", "4783fe10ac3adce15ac8f358ef5462739852c569"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticate_model_files(model_dir: Path) -> list[dict]:
    records = []
    for filename, (size, algorithm, expected) in sorted(MODEL_FILES.items()):
        path = model_dir / filename
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"Missing or wrong-sized pinned model file: {filename}")
        sha256 = hashlib.sha256()
        git_sha1 = hashlib.sha1(f"blob {size}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                sha256.update(block)
                if algorithm == "git_blob_sha1":
                    git_sha1.update(block)
        observed = git_sha1.hexdigest() if algorithm == "git_blob_sha1" else sha256.hexdigest()
        if observed != expected:
            raise ValueError(f"Pinned model file digest mismatch: {filename}")
        records.append({"file": filename, "size_bytes": size, "sha256": sha256.hexdigest(),
                        "upstream_digest_type": algorithm, "upstream_digest": expected})
    return records


def read_genes(path: Path) -> list[str]:
    """Read a named gene column or a single headerless column; stable deduplication."""
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = [row for row in csv.reader(stream) if row and any(v.strip() for v in row)]
    if not rows:
        raise ValueError("Gene CSV is empty")
    headers = [value.strip().lower() for value in rows[0]]
    aliases = ("gene", "gene_name", "gene_symbol", "target_gene", "symbol")
    column = next((headers.index(name) for name in aliases if name in headers), None)
    if column is None:
        if len(rows[0]) != 1:
            raise ValueError("Multi-column CSV requires a named gene column")
        column = 0
        data = rows
    else:
        data = rows[1:]
    genes = []
    seen = set()
    for row in data:
        if column >= len(row):
            raise ValueError("CSV row lacks its gene column")
        gene = row[column].strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", gene):
            raise ValueError(f"Invalid gene symbol: {gene!r}")
        if gene not in seen:
            genes.append(gene)
            seen.add(gene)
    if not genes:
        raise ValueError("Gene CSV contains no genes")
    return genes


def make_prompts(genes: list[str]) -> list[str]:
    return [PROMPT_TEMPLATE.format(gene=gene) for gene in genes]


def pool_hidden(hidden, attention_mask):
    """FP32 mean over non-padding text tokens, followed by row L2 normalization."""
    import torch

    if hidden.ndim != 3 or attention_mask.shape != hidden.shape[:2]:
        raise ValueError("Hidden-state and attention-mask shapes do not match")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("Attention mask must be binary")
    mask = attention_mask.to(dtype=torch.float32).unsqueeze(-1)
    lengths = mask.sum(dim=1)
    if torch.any(lengths == 0):
        raise ValueError("Cannot pool an empty token sequence")
    # masked_fill excludes even NaNs in padded positions from the mean.
    masked = hidden.float().masked_fill(~attention_mask.bool().unsqueeze(-1), 0.0)
    pooled = masked.sum(dim=1) / lengths
    norms = pooled.norm(p=2, dim=1, keepdim=True)
    if not torch.isfinite(pooled).all() or not torch.isfinite(norms).all() or torch.any(norms <= 0):
        raise ValueError("Lingshu produced a non-finite or zero feature vector")
    return pooled / norms


def extract_batches(model, tokenizer, genes: list[str], batch_size: int, device) -> np.ndarray:
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    vectors = []
    model.eval()
    model.requires_grad_(False)
    with torch.inference_mode():
        for start in range(0, len(genes), batch_size):
            encoded = tokenizer(make_prompts(genes[start:start + batch_size]),
                                padding=True, truncation=False, add_special_tokens=False,
                                return_tensors="pt")
            if encoded["input_ids"].shape[1] > MAX_TOKENS:
                raise ValueError("Gene prompt exceeds registered token limit; truncation is forbidden")
            inputs = {key: encoded[key].to(device) for key in ("input_ids", "attention_mask")}
            # Calling the base model omits the large vocabulary-logit allocation.
            hidden = model.model(**inputs, use_cache=False, output_hidden_states=False,
                                 return_dict=True).last_hidden_state
            pooled = pool_hidden(hidden, inputs["attention_mask"])
            if pooled.shape[1] != EMBEDDING_DIMENSION:
                raise ValueError("Unexpected Lingshu hidden dimension")
            vectors.append(pooled.cpu().numpy().astype(np.float32, copy=False))
    return np.ascontiguousarray(np.concatenate(vectors, axis=0), dtype=np.float32)


def publish_bytes(path: Path, payload: bytes) -> None:
    """Atomically create one file, never replace even a racing writer's output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        os.unlink(temporary)


def deterministic_npz(genes: list[str], embeddings: np.ndarray) -> bytes:
    if embeddings.dtype != np.float32 or embeddings.shape != (len(genes), EMBEDDING_DIMENSION):
        raise ValueError("Embedding output shape or dtype mismatch")
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, array in (("gene_names", np.asarray(genes, dtype=np.str_)),
                            ("embeddings", embeddings.astype("<f4", copy=False))):
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.external_attr = 0o600 << 16
            archive.writestr(entry, buffer.getvalue())
    return output.getvalue()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--genes-csv", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--require-h100", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        raise ValueError("batch_size must be positive")
    if args.output_npz.resolve() == args.output_json.resolve():
        raise ValueError("NPZ and JSON must have different output paths")
    for path in (args.output_npz, args.output_json):
        if os.path.lexists(path):
            raise FileExistsError(f"Refusing to overwrite output: {path}")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError(f"Require transformers=={TRANSFORMERS_VERSION}")
    if not torch.cuda.is_available():
        raise RuntimeError("Extraction requires an allocated CUDA GPU")
    gpu_name = torch.cuda.get_device_name(0)
    if args.require_h100 and ("H100" not in gpu_name or not os.environ.get("SLURM_JOB_ID")):
        raise RuntimeError("--require-h100 requires an H100 inside a Slurm allocation")
    torch.manual_seed(20260909)
    torch.cuda.manual_seed_all(20260909)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    genes = read_genes(args.genes_csv)
    source_files = authenticate_model_files(args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True,
                                             trust_remote_code=False, use_fast=True)
    tokenizer.padding_side = "right"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="eager", device_map={"": "cuda:0"},
        use_safetensors=True,
    )
    embeddings = extract_batches(model, tokenizer, genes, args.batch_size, torch.device("cuda:0"))
    payload = deterministic_npz(genes, embeddings)
    receipt = {
        "schema": SCHEMA, "status": "complete", "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION, "model_license": "MIT",
        "role": "frozen_symbol_prompt_language_features", "perturbation_response_training": False,
        "challenge_treated_data_used": False, "generated_annotations_used": False,
        "gene_count": len(genes), "embedding_dimension": EMBEDDING_DIMENSION,
        "gene_names": genes, "model_files": source_files,
        "genes_csv_sha256": sha256_file(args.genes_csv),
        "prompt_template": PROMPT_TEMPLATE,
        "ordered_prompts_sha256": hashlib.sha256(json.dumps(make_prompts(genes), ensure_ascii=False,
                                                            separators=(",", ":")).encode()).hexdigest(),
        "pooling": "last_hidden_state_attention_masked_mean_float32_then_l2",
        "tokenization": {"chat_template": False, "add_special_tokens": False,
                         "padding_side": "right", "truncation": False,
                         "maximum_accepted_tokens": MAX_TOKENS},
        "inference": {"weights_dtype": "bfloat16", "output_dtype": "float32",
                      "batch_size": args.batch_size, "attention": "eager",
                      "seed": 20260909, "deterministic_algorithms": True,
                      "model_eval": True, "gradient_enabled": False},
        "runtime": {"hostname": socket.gethostname(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "gpu": gpu_name, "cuda": torch.version.cuda,
                    "versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "tokenizers", "numpy", "safetensors")}},
        "extractor_sha256": sha256_file(Path(__file__)),
        "npz": {"path": str(args.output_npz.resolve()), "size_bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "keys": ["gene_names", "embeddings"]},
    }
    publish_bytes(args.output_npz, payload)
    publish_bytes(args.output_json, (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"status": "complete", "gene_count": len(genes),
                      "embedding_dimension": EMBEDDING_DIMENSION, "npz_sha256": receipt["npz"]["sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
