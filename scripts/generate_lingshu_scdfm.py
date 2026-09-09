"""Generate VCC raw counts from a frozen public-data Lingshu/scDFM adapter.

Only released controls, target identifiers, the gene axis, and frozen trained
artifacts are accepted. Unmodeled genes copy the sampled controls exactly.
Modeled counts use inverse CP10k normalization and stochastic rounding. There
is no manual knockdown, gene-specific override, or challenge-treated input.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import anndata as ad
from anndata.io import write_elem
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
import torch

from lingshu_scdfm_model import (
    CHECKPOINT_SCHEMA, LingshuScDFM, atomic_json, authenticate_upstream,
    integrate, load_embeddings, require, sha256_file,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--training-summary", type=Path)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--controls", action="append", required=True, metavar="CONTEXT=FILE")
    p.add_argument("--gene-axis", type=Path, required=True)
    p.add_argument("--targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cells-per-target", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--ode-steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--cpu", action="store_true", help="Small local integration tests only")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--test-subset", action="store_true", help="Small non-submission integration test; marked in receipt")
    return p


def read_axis(path):
    frame = pd.read_csv(path, dtype=str)
    require(frame.shape[1] == 1, f"Expected single-column axis: {path}")
    values = frame.iloc[:, 0].tolist()
    require(values and all(isinstance(value, str) and value for value in values), "Empty axis value")
    require(len(set(values)) == len(values), "Duplicate axis values")
    return values


def replace_modeled_counts(raw, modeled_log, gene_indices, library_sizes, rng):
    """Inverse-normalize learned values; preserve every unmodeled sparse entry."""
    raw = sparse.csr_matrix(raw)
    require(np.isfinite(raw.data).all() and (raw.data >= 0).all(), "Controls are not nonnegative counts")
    require(np.equal(raw.data, np.rint(raw.data)).all(), "Controls must contain raw integer counts")
    values = np.asarray(modeled_log, dtype=np.float64)
    require(values.shape == (raw.shape[0], len(gene_indices)), "Modeled output shape mismatch")
    require(np.isfinite(values).all(), "Model output is not finite")
    require(np.isfinite(library_sizes).all() and (library_sizes > 0).all(), "Invalid normalization library")
    # This is the fixed count decoder's mathematical log-CP10k domain, not a
    # target-dependent effect adjustment. Report the clamp fraction in receipt.
    bounded = np.clip(values, 0.0, np.log1p(10000.0))
    expected = np.expm1(bounded) * np.asarray(library_sizes)[:, None] / 10000.0
    require(np.max(expected, initial=0) < np.iinfo(np.int32).max, "Count overflow")
    rounded = (np.floor(expected) + (rng.random(expected.shape) < (expected % 1))).astype(np.int32)
    modeled = sparse.csr_matrix(rounded)
    modeled.indices = np.asarray(gene_indices, dtype=np.int32)[modeled.indices]
    modeled._shape = raw.shape
    kept = raw.astype(np.int32, copy=True)
    kept.data[np.isin(kept.indices, gene_indices)] = 0
    kept.eliminate_zeros()
    result = (kept + modeled).tocsr()
    result.sort_indices()
    require(result.nnz == 0 or int(result.data.min()) > 0, "Noncanonical emitted counts")
    return result, {"clamped_low": int((values < 0).sum()),
                    "clamped_high": int((values > np.log1p(10000.0)).sum()),
                    "modeled_values": int(values.size)}


class StreamingH5AD:
    """Append integer CSR chunks, then write normal AnnData axis metadata."""
    def __init__(self, path, rows, genes):
        self.handle = h5py.File(path, "x")
        self.handle.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        self.group = self.handle.create_group("X")
        self.group.attrs.update({"encoding-type": "csr_matrix", "encoding-version": "0.1.0", "shape": (rows, genes)})
        options = {"shape": (0,), "maxshape": (None,), "chunks": (262144,), "compression": "lzf"}
        self.values = self.group.create_dataset("data", dtype="int32", **options)
        self.indices = self.group.create_dataset("indices", dtype="int32", **options)
        self.indptr = self.group.create_dataset("indptr", shape=(1,), maxshape=(None,), dtype="int64", chunks=(65536,))
        self.indptr[0] = 0
        self.rows, self.nnz, self.expected_rows = 0, 0, rows

    def append(self, block):
        block = sparse.csr_matrix(block, dtype=np.int32)
        for destination, values in ((self.values, block.data), (self.indices, block.indices)):
            destination.resize((self.nnz + block.nnz,))
            destination[self.nnz:] = values
        self.indptr.resize((self.rows + block.shape[0] + 1,))
        self.indptr[self.rows + 1:] = block.indptr[1:].astype(np.int64) + self.nnz
        self.rows += block.shape[0]
        self.nnz += block.nnz

    def finish(self, obs, var, uns):
        require(self.rows == self.expected_rows == len(obs), "Incomplete prediction rows")
        write_elem(self.handle, "obs", obs)
        write_elem(self.handle, "var", var)
        write_elem(self.handle, "uns", uns)
        for name in ("obsm", "varm", "obsp", "varp", "layers"):
            write_elem(self.handle, name, {})
        self.handle.flush()

    def close(self):
        self.handle.close()


def generate(args):
    require(args.batch_size > 0 and args.ode_steps > 0 and args.cells_per_target > 0, "Invalid generation dimensions")
    if not args.cpu:
        require(bool(os.environ.get("SLURM_JOB_ID")), "GPU generation must run inside Slurm")
        require(torch.cuda.is_available(), "Allocated CUDA GPU unavailable")
    device = torch.device("cpu" if args.cpu else "cuda")
    bf16 = device.type == "cuda" and not args.no_bf16
    torch.set_num_threads(max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)))
    require(not args.output.exists(), "Prediction output exists")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = args.training_summary or args.checkpoint.parent / "training_summary.json"
    summary = json.loads(summary_path.read_text())
    checkpoint_sha = sha256_file(args.checkpoint)
    require(summary.get("status") == "completed" and summary.get("best_checkpoint_sha256") == checkpoint_sha,
            "Checkpoint is not the completed training run's selected best")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    require(checkpoint.get("schema") == CHECKPOINT_SCHEMA and checkpoint.get("challenge_treated_used") is False,
            "Checkpoint is not a public-data Lingshu/scDFM fit")
    require(not checkpoint["training_config"].get("debug_test_features") or args.test_subset,
            "Synthetic-test checkpoint cannot create an official prediction")
    require(checkpoint["code_sha256"]["lingshu_scdfm_model.py"] == sha256_file(Path(__file__).parent / "lingshu_scdfm_model.py"),
            "Model implementation changed after training")
    require(checkpoint["step"] == checkpoint["best_step"] == summary["best_step"], "Selected step is inconsistent")
    require(checkpoint["validation"]["selection_velocity_mse"] == summary["best_score"], "Selected validation score differs")
    embeddings, embeddings_sha = load_embeddings(args.embeddings)
    require(embeddings_sha == checkpoint["embeddings_sha256"], "Lingshu embeddings differ from training")
    require(authenticate_upstream(args.source) == checkpoint["upstream"], "Upstream PAD changed")
    targets, gene_names = read_axis(args.targets), read_axis(args.gene_axis)
    require("non-targeting" not in targets, "Targets contain control label")
    controls = {}
    for item in args.controls:
        context, separator, location = item.partition("=")
        require(separator == "=" and context and location and context not in controls, "Controls must be unique CONTEXT=FILE entries")
        controls[context] = Path(location)
    if not args.test_subset:
        require(set(controls) == {"A", "B", "C"} and len(targets) == 300 and len(gene_names) == 18533
                and args.cells_per_target == 400, "Official VCC dimensions/context set differ")
    require(set(targets) <= set(embeddings), f"Missing target embeddings: {sorted(set(targets) - set(embeddings))}")
    selected_genes = np.asarray(checkpoint["gene_indices"], dtype=np.int64)
    require(selected_genes.max() < len(gene_names) and [gene_names[index] for index in selected_genes] == checkpoint["gene_names"],
            "Checkpoint modeled genes do not match official axis")
    require(checkpoint["metadata"]["normalization"].get("full_axis_scope") == "shared_public_challenge_gene_axis",
            "Checkpoint did not register a shared normalization axis")
    normalization_names = checkpoint["normalization_gene_names"]
    require(normalization_names and len(set(normalization_names)) == len(normalization_names), "Invalid normalization axis")
    gene_lookup = {name: index for index, name in enumerate(gene_names)}
    require(set(normalization_names) <= set(gene_lookup), "Normalization axis has genes absent from controls")
    normalization_indices = np.asarray([gene_lookup[name] for name in normalization_names], dtype=np.int64)
    model = LingshuScDFM(checkpoint["model_config"], args.source).to(device)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.eval()
    del checkpoint["optimizer_state"]
    expected_rows = len(controls) * len(targets) * args.cells_per_target
    partial = args.output.with_name(args.output.name + ".partial")
    require(not partial.exists(), "Incomplete prediction exists; use a new output path")
    writer = StreamingH5AD(partial, expected_rows, len(gene_names))
    rows_context, rows_target, rows_source, rows_name = [], [], [], []
    counts = {"clamped_low": 0, "clamped_high": 0, "modeled_values": 0}
    group_receipts, control_descriptors = [], {}
    try:
        for context, control_path in sorted(controls.items()):
            require(control_path.is_file(), f"Missing released controls: {control_path}")
            control_descriptors[context] = {"path": str(control_path.resolve()), "sha256": sha256_file(control_path)}
            control = ad.read_h5ad(control_path, backed="r")
            try:
                require(control.var_names.astype(str).tolist() == gene_names, "Released control gene axis differs")
                require("target_gene" in control.obs and set(control.obs["target_gene"].astype(str)) == {"non-targeting"},
                        "Generator accepts controls-only expression")
                if "context" in control.obs:
                    require(set(control.obs["context"].astype(str)) == {context}, "Control context mismatch")
                require(control.n_obs >= args.cells_per_target, "Not enough released control cells")
                for target in targets:
                    target_seed = int.from_bytes(hashlib.sha256(f"{args.seed}|{context}|{target}".encode()).digest()[:8], "little")
                    rng = np.random.default_rng(target_seed)
                    torch_rng = torch.Generator(device=device).manual_seed(target_seed)
                    sampled = rng.choice(control.n_obs, args.cells_per_target, replace=False)
                    input_sum, output_sum = 0, 0
                    for start in range(0, len(sampled), args.batch_size):
                        selected = sampled[start:start + args.batch_size]
                        order = np.argsort(selected)
                        raw = sparse.csr_matrix(control.X[selected[order]])[np.argsort(order)]
                        library = np.asarray(raw[:, normalization_indices].sum(axis=1)).ravel().astype(np.float64)
                        require((library > 0).all(), "Empty control library")
                        normalized = np.log1p(raw[:, selected_genes].toarray().astype(np.float64) * (10000.0 / library[:, None])).astype(np.float32)
                        control_tensor = torch.from_numpy(normalized).to(device)
                        feature_tensor = torch.from_numpy(np.repeat(embeddings[target][None, :], len(selected), axis=0)).to(device)
                        output = integrate(model, control_tensor, feature_tensor, steps=args.ode_steps,
                                           noise_std=checkpoint["training_config"]["noise_std"], generator=torch_rng, bf16=bf16)
                        block, stats = replace_modeled_counts(raw, output.cpu().numpy(), selected_genes, library, rng)
                        writer.append(block)
                        input_sum += int(raw.sum())
                        output_sum += int(block.sum())
                        for key, value in stats.items():
                            counts[key] += value
                        rows_context.extend([context] * len(selected))
                        rows_target.extend([target] * len(selected))
                        rows_source.extend(selected.tolist())
                        rows_name.extend([f"{context}|{target}|{start + index:04d}" for index in range(len(selected))])
                    group = {"context": context, "target_gene": target, "input_library_sum": input_sum,
                             "output_library_sum": output_sum, "library_sum_ratio": output_sum / max(input_sum, 1),
                             "sampled_control_rows_sha256": hashlib.sha256(sampled.astype("<i8").tobytes()).hexdigest()}
                    group_receipts.append(group)
                    print(json.dumps({"context": context, "target": target, "rows_written": writer.rows,
                                      "library_sum_ratio": group["library_sum_ratio"]}), flush=True)
            finally:
                control.file.close()
        obs = pd.DataFrame({"context": pd.Categorical(rows_context), "target_gene": pd.Categorical(rows_target),
                            "source_control_row": rows_source}, index=pd.Index(rows_name, name="cell_id"))
        var = pd.DataFrame(index=pd.Index(gene_names, name="gene_name"))
        writer.finish(obs, var, {"lingshu_scdfm": {"schema": "vcc-lingshu-scdfm-generated-counts-v1",
                      "checkpoint_sha256": checkpoint_sha, "seed": args.seed,
                      "challenge_treated_used": False, "test_subset": args.test_subset}})
        nnz = writer.nnz
    finally:
        writer.close()
    os.replace(partial, args.output)
    result = {"schema": "vcc-lingshu-scdfm-generation-receipt-v1", "status": "completed",
              "prediction_sha256": sha256_file(args.output), "shape": [expected_rows, len(gene_names)], "nnz": nnz,
              "checkpoint_sha256": checkpoint_sha, "training_summary_sha256": sha256_file(summary_path),
              "embeddings_sha256": embeddings_sha, "gene_axis_sha256": sha256_file(args.gene_axis),
              "targets_sha256": sha256_file(args.targets), "controls": control_descriptors, "seed": args.seed,
              "ode_solver": "euler", "ode_steps": args.ode_steps, "batch_size": args.batch_size,
              "normalization_gene_count": len(normalization_names), "modeled_gene_count": len(selected_genes),
              "unmodeled_gene_policy": "copy_sampled_control_integer_counts_exactly",
              "count_decoder": "clip_log_cp10k_domain_then_inverse_normalize_and_stochastic_round",
              "manual_target_effects": False, "clamping": counts, "group_receipts": group_receipts,
              "challenge_treated_used": False, "test_subset": args.test_subset,
              "fit_scope": "public_source_data_only_zero_shot_for_challenge", "biological_quality_validated": False,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "code_sha256": {name: sha256_file(Path(__file__).parent / name) for name in ("generate_lingshu_scdfm.py", "lingshu_scdfm_model.py")}}
    atomic_json(args.output.with_suffix(args.output.suffix + ".receipt.json"), result)
    return result


if __name__ == "__main__":
    print(json.dumps(generate(parser().parse_args()), indent=2))
