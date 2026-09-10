"""Train a compact official scDFM PAD with frozen Lingshu target features.

Training reads a prepared public-data cache only. Checkpoint selection averages
whole-context and whole-target validation CFM errors equally. Neither challenge
perturbed cells nor a leaderboard score is an input.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from lingshu_scdfm_model import (
    CHECKPOINT_SCHEMA, LingshuScDFM, atomic_json, authenticate_upstream, authenticate_embedding_receipt,
    grouped_rows, integrate, load_cache, load_embeddings, mmd2_unbiased, require, sha256_file,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--embeddings", type=Path, required=True)
    p.add_argument("--embedding-receipt", type=Path)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--validate-every", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--validation-batch-size", type=int, default=16)
    p.add_argument("--max-validation-groups", type=int, default=32,
                   help="Deterministic evenly spaced groups per validation kind")
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--mmd-weight", type=float, default=0.5)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--resume", type=Path)
    p.add_argument("--cpu", action="store_true", help="Small local unit/integration tests only")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--debug-test-features", action="store_true", help="CPU synthetic integration tests only; never a production checkpoint")
    return p


def validation_groups(data, limit):
    all_groups = grouped_rows(data["val_targets"], data["val_contexts"], data["val_kind"])
    chosen = []
    for kind in ("context", "target"):
        groups = [(key, rows) for key, rows in all_groups if key[0] == kind]
        require(groups, f"No {kind} validation groups")
        indices = np.unique(np.linspace(0, len(groups) - 1, min(len(groups), limit), dtype=int))
        chosen.extend(groups[index] for index in indices)
    return chosen


@torch.inference_mode()
def validate(model, data, embedding_map, groups, *, device, batch_size, noise_std, seed, bf16):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    metrics = {"context": [], "target": []}
    baseline = {"context": [], "target": []}
    group_scores = []
    for key, rows in groups:
        selected = rows[np.arange(batch_size) % len(rows)]
        target = torch.from_numpy(data["val_x"][selected]).to(device)
        control = torch.from_numpy(data["val_control"][selected]).to(device)
        features = torch.from_numpy(np.stack([embedding_map[name] for name in data["val_targets"][selected]])).to(device)
        x0 = control + torch.randn(control.shape, generator=generator, device=device) * noise_std
        t = torch.linspace(0.05, 0.95, batch_size, device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * target
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity = model(xt, control, features, t)
        value = float(torch.nn.functional.mse_loss(velocity.float(), target - x0))
        require(math.isfinite(value), "Nonfinite validation error")
        metrics[key[0]].append(value)
        control_error = float(torch.mean((target - x0).square()))
        baseline[key[0]].append(control_error)
        group_scores.append({"kind": key[0], "context": key[1], "target": key[2], "velocity_mse": value,
                             "zero_velocity_mse": control_error})
    means = {kind: float(np.mean(scores)) for kind, scores in metrics.items()}
    baseline_means = {kind: float(np.mean(scores)) for kind, scores in baseline.items()}
    return {"selection_velocity_mse": float(np.mean(list(means.values()))),
            "kind_velocity_mse": means, "kind_zero_velocity_mse": baseline_means,
            "improves_zero_velocity_each_kind": all(means[kind] < baseline_means[kind] for kind in means),
            "groups": group_scores}


def save_checkpoint(path, payload):
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.inference_mode()
def rollout_diagnostics(model, data, embedding_map, groups, *, device, batch_size, noise_std, seed, bf16):
    """Evaluate the already selected model; never choose checkpoints here."""
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed)
    scores = []
    for key, rows in groups:
        selected = rows[np.arange(batch_size) % len(rows)]
        target = torch.from_numpy(data["val_x"][selected]).to(device)
        control = torch.from_numpy(data["val_control"][selected]).to(device)
        features = torch.from_numpy(np.stack([embedding_map[name] for name in data["val_targets"][selected]])).to(device)
        raw_endpoint = integrate(model, control, features, steps=20, noise_std=noise_std,
                                 generator=generator, bf16=bf16)
        endpoint = raw_endpoint.clamp(0.0, math.log1p(10000.0))
        target_mean = target.mean(dim=0)
        score = {"kind": key[0], "context": key[1], "target": key[2],
                 "model_centroid_mse": float((endpoint.mean(dim=0) - target_mean).square().mean()),
                 "control_centroid_mse": float((control.mean(dim=0) - target_mean).square().mean()),
                 "model_mmd2": float(mmd2_unbiased(endpoint, target)),
                 "control_mmd2": float(mmd2_unbiased(control, target)),
                 "clamped_low_fraction": float((raw_endpoint < 0).float().mean()),
                 "clamped_high_fraction": float((raw_endpoint > math.log1p(10000.0)).float().mean())}
        require(all(math.isfinite(value) for value in score.values() if isinstance(value, float)), "Nonfinite public rollout diagnostic")
        scores.append(score)
    metrics = ("model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
               "clamped_low_fraction", "clamped_high_fraction")
    kinds = {kind: {metric: float(np.mean([row[metric] for row in scores if row["kind"] == kind]))
                   for metric in metrics} for kind in ("context", "target")}
    return {"role": "diagnostic_only_after_checkpoint_selection", "checkpoint_selection_affected": False,
            "expression_space": "log1p_cp10k_shared_gene_axis_modeled_panel", "ode_solver": "euler",
            "ode_steps": 20, "cells_per_group": batch_size, "seed": seed,
            "count_decoder_domain_clamp_applied": True, "unpaired_cells_compared_by_centroid_and_mmd": True,
            "kind_metrics": kinds, "groups": scores,
            "improves_control_centroid_mse_each_kind": all(row["model_centroid_mse"] < row["control_centroid_mse"] for row in kinds.values()),
            "improves_control_mmd2_each_kind": all(row["model_mmd2"] < row["control_mmd2"] for row in kinds.values()),
            "official_challenge_metrics_evaluated": False, "challenge_treated_used": False}


def carry_forward_best(resume_path, output_dir, resumed):
    """Keep an earlier selected checkpoint when a resumed run never improves."""
    original = resume_path.parent / f"step_{resumed['best_step']:07d}.pt"
    require(original.is_file(), "Resume is missing its previously selected immutable checkpoint")
    payload = torch.load(original, map_location="cpu", weights_only=True)
    for key in ("schema", "cache_sha256", "embeddings_sha256", "model_config", "training_config", "upstream"):
        require(payload[key] == resumed[key], f"Previously selected checkpoint changed {key}")
    require(payload["step"] == resumed["best_step"] and
            payload["validation"]["selection_velocity_mse"] == resumed["best_score"], "Previously selected checkpoint has wrong step/score")
    # Carry the immutable archive as well, so another resume from the new
    # directory remains self-contained even if the old best never improves.
    for destination in (output_dir / original.name, output_dir / "best.pt"):
        if destination.exists():
            require(sha256_file(destination) == sha256_file(original), "Existing best checkpoint disagrees with resume selection")
        else:
            temporary = destination.with_name(destination.name + f".tmp.{os.getpid()}")
            shutil.copyfile(original, temporary)
            os.replace(temporary, destination)


def train(args):
    require(args.steps > 0 and args.validate_every > 0 and args.batch_size >= 2, "Invalid training length or batch")
    require(args.validation_batch_size >= 2 and args.max_validation_groups > 0, "Invalid validation batch")
    require(args.learning_rate > 0 and args.mmd_weight >= 0 and args.noise_std >= 0, "Invalid optimizer/flow configuration")
    require(not args.debug_test_features or args.cpu, "Synthetic feature bypass is restricted to CPU tests")
    if not args.cpu:
        require(bool(os.environ.get("SLURM_JOB_ID")), "GPU training must run in a Slurm allocation")
        require(torch.cuda.is_available(), "Allocated CUDA GPU unavailable")
        require("H100" in torch.cuda.get_device_name(0), "This launcher is registered for H100")
    device = torch.device("cpu" if args.cpu else "cuda")
    bf16 = device.type == "cuda" and not args.no_bf16
    torch.set_num_threads(max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)))
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and args.resume is None:
        raise ValueError("Output directory already contains artifacts; use a new directory or explicit resume")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data, cache_sha = load_cache(args.cache)
    embedding_map, embeddings_sha = load_embeddings(args.embeddings)
    if args.debug_test_features:
        embedding_receipt = {"synthetic_test_only": True}
    else:
        require(args.embedding_receipt is not None, "Production training requires a Lingshu extraction receipt")
        embedding_receipt = authenticate_embedding_receipt(args.embedding_receipt, args.embeddings, embedding_map, embeddings_sha)
    requested = set(data["train_targets"].tolist()) | set(data["val_targets"].tolist())
    require(requested <= set(embedding_map), f"Missing Lingshu target features: {sorted(requested - set(embedding_map))}")
    source = authenticate_upstream(args.source)
    model_config = {"gene_count": len(data["gene_names"]), "embedding_dimension": len(next(iter(embedding_map.values()))),
                    "hidden_size": args.hidden_size, "heads": args.heads, "layers": args.layers, "dropout": args.dropout,
                    "backbone": "official_scdfm_differential_perceiver", "gene_correlation_graph": False,
                    "initialization": "from_scratch", "embedding_normalization": "per_vector_l2"}
    training_config = {key: getattr(args, key) for key in ("batch_size", "validation_batch_size", "max_validation_groups",
                       "learning_rate", "mmd_weight", "noise_std", "seed")}
    training_config["bf16"] = bf16
    training_config["debug_test_features"] = args.debug_test_features
    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    model = LingshuScDFM(model_config, args.source).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4)
    train_groups = grouped_rows(data["train_targets"], data["train_contexts"])
    val_groups = validation_groups(data, args.max_validation_groups)
    code_hashes = {name: sha256_file(Path(__file__).parent / name) for name in ("train_lingshu_scdfm.py", "lingshu_scdfm_model.py")}
    start_step, best_score, best_step = 0, float("inf"), None
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        require(checkpoint.get("schema") == CHECKPOINT_SCHEMA, "Invalid resume checkpoint")
        for name, expected in (("cache_sha256", cache_sha), ("embeddings_sha256", embeddings_sha),
                               ("model_config", model_config), ("training_config", training_config), ("upstream", source),
                               ("code_sha256", code_hashes), ("embedding_receipt", embedding_receipt)):
            require(checkpoint[name] == expected, f"Resume changed {name}")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        np_rng.bit_generator.state = checkpoint["numpy_rng_state"]
        start_step, best_score, best_step = checkpoint["step"], checkpoint["best_score"], checkpoint["best_step"]
        carry_forward_best(args.resume, args.output_dir, checkpoint)
    provenance = {"schema": "vcc-lingshu-scdfm-training-provenance-v1", "upstream": source,
                  "cache_sha256": cache_sha, "embeddings_sha256": embeddings_sha, "model_config": model_config,
                  "training_config": training_config, "cache_metadata": data["metadata"], "code_sha256": code_hashes,
                  "challenge_treated_used": False, "train_group_count": len(train_groups),
                  "validation_group_count": len(val_groups), "checkpoint_selection": "earliest_minimum_equal_kind_mean_heldout_velocity_mse",
                  "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "device": str(device),
                  "torch_version": str(torch.__version__), "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    provenance["embedding_receipt"] = embedding_receipt
    if args.embedding_receipt:
        provenance["embedding_receipt_sha256"] = sha256_file(args.embedding_receipt)
    atomic_json(args.output_dir / "training_provenance.json", provenance)
    started = time.monotonic()
    require(start_step < args.steps, "Resume checkpoint already reached requested steps")
    for step in range(start_step + 1, args.steps + 1):
        model.train()
        _, rows = train_groups[int(np_rng.integers(len(train_groups)))]
        selected = np_rng.choice(rows, args.batch_size, replace=len(rows) < args.batch_size)
        target = torch.from_numpy(data["train_x"][selected]).to(device)
        control = torch.from_numpy(data["train_control"][selected]).to(device)
        features = torch.from_numpy(np.stack([embedding_map[name] for name in data["train_targets"][selected]])).to(device)
        x0 = control + torch.randn_like(control) * args.noise_std
        t = torch.rand(args.batch_size, device=device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * target
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity = model(xt, control, features, t)
        velocity = velocity.float()
        cfm = torch.nn.functional.mse_loss(velocity, target - x0)
        endpoint = xt + (1 - t[:, None]) * velocity
        mmd = mmd2_unbiased(endpoint, target)
        loss = cfm + args.mmd_weight * mmd
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "cfm": float(cfm.detach()), "mmd": float(mmd.detach()),
                              "loss": float(loss.detach()), "gradient_norm": float(norm),
                              "elapsed_seconds": round(time.monotonic() - started, 2)}), flush=True)
        if step % args.validate_every != 0 and step != args.steps:
            continue
        metrics = validate(model, data, embedding_map, val_groups, device=device,
                           batch_size=args.validation_batch_size, noise_std=args.noise_std,
                           seed=args.seed + 991, bf16=bf16)
        score = metrics["selection_velocity_mse"]
        improved = score < best_score
        if improved:
            best_score, best_step = score, step
        payload = {"schema": CHECKPOINT_SCHEMA, "step": step, "best_step": best_step, "best_score": best_score,
                   "validation": metrics, "model_config": model_config, "training_config": training_config,
                   "cache_sha256": cache_sha, "embeddings_sha256": embeddings_sha, "upstream": source,
                   "code_sha256": code_hashes, "gene_names": data["gene_names"].tolist(),
                   "embedding_receipt": embedding_receipt,
                   "gene_indices": data["gene_indices"].tolist(), "metadata": data["metadata"],
                   "normalization_gene_names": data["normalization_gene_names"].tolist(),
                   "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                   "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   "numpy_rng_state": np_rng.bit_generator.state, "challenge_treated_used": False}
        archive = args.output_dir / f"step_{step:07d}.pt"
        require(not archive.exists(), f"Refusing to overwrite checkpoint archive: {archive}")
        save_checkpoint(archive, payload)
        save_checkpoint(args.output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        atomic_json(args.output_dir / f"validation_{step:07d}.json", metrics)
        print(json.dumps({"step": step, "validation_velocity_mse": score, "best_step": best_step}), flush=True)
    selected_checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected_checkpoint["model_state"], strict=True)
    deployment_diagnostics = rollout_diagnostics(model, data, embedding_map, val_groups, device=device,
                                                 batch_size=args.validation_batch_size, noise_std=args.noise_std,
                                                 seed=args.seed + 1771, bf16=bf16)
    atomic_json(args.output_dir / "best_rollout_diagnostics.json", deployment_diagnostics)
    summary = {"schema": "vcc-lingshu-scdfm-training-summary-v1", "status": "completed", "last_step": args.steps,
               "best_step": best_step, "best_score": best_score, "best_checkpoint_sha256": sha256_file(args.output_dir / "best.pt"),
               "training_provenance_sha256": sha256_file(args.output_dir / "training_provenance.json"),
               "elapsed_seconds": time.monotonic() - started, "challenge_treated_used": False,
               "public_model_quality_evaluated": True, "official_challenge_metrics_evaluated": False,
               "validation_metric_scope": "heldout_velocity_selection_then_diagnostic_endpoint_rollout",
               "deployment_rollout_diagnostics": deployment_diagnostics}
    atomic_json(args.output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
