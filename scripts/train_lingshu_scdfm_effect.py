"""Fit control-anchored public group effects; keep challenge-treated cells unseen.

Checkpoint selection uses zero-start Euler predictions against all-cell group
means on fixed public development groups. The subsequent sampled-cell deployment
diagnostics reuse these development groups and are NOT an independent test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from lingshu_scdfm_model import (
    atomic_json, authenticate_embedding_receipt, authenticate_upstream, load_cache,
    load_embeddings, require, sha256_file,
)
from lingshu_scdfm_effect_model import (
    CHECKPOINT_SCHEMA, METHOD, LingshuScDFMEffect, feature_mapping,
    fit_feature_normalization, group_effects, predict_group_effects,
)
from train_lingshu_scdfm import rollout_diagnostics, save_checkpoint, validation_groups


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for argument in ("cache", "embeddings", "source", "output-dir"):
        p.add_argument(f"--{argument}", type=Path, required=True)
    p.add_argument("--embedding-receipt", type=Path)
    p.add_argument("--feature-mode", choices=("true", "shuffled", "constant"), default="true")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--validate-every", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--validation-batch-size", type=int, default=16)
    p.add_argument("--max-validation-groups", type=int, default=32)
    p.add_argument("--hidden-size", type=int, default=128)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260909)
    p.add_argument("--cpu", action="store_true", help="Local synthetic integration tests only")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--debug-test-features", action="store_true", help="CPU synthetic tests only; never production")
    return p


def training_times(batch_size, device):
    """Half zero-start conditioning supervision; half ordinary uniform CFM."""
    require(batch_size >= 2 and batch_size % 2 == 0, "An even training batch is required")
    return torch.cat((torch.zeros(batch_size // 2, device=device),
                      torch.rand(batch_size // 2, device=device)))


@torch.inference_mode()
def validate_group_rollouts(model, groups, effects, embedding_map, *, device, batch_size, bf16):
    """Selection metric is an actual rollout, not a teacher-forced velocity."""
    model.eval()
    scores = []
    for start in range(0, len(groups), batch_size):
        selected = groups[start:start + batch_size]
        features = torch.from_numpy(np.stack([embedding_map[key[-1]] for key, _ in selected])).to(device)
        prediction = predict_group_effects(model, features, gene_count=effects.shape[1], steps=20, bf16=bf16)
        truth = torch.from_numpy(effects[start:start + len(selected)]).to(device)
        errors = (prediction - truth).square().mean(dim=1).cpu().tolist()
        baselines = truth.square().mean(dim=1).cpu().tolist()
        for (key, rows), error, baseline in zip(selected, errors, baselines, strict=True):
            require(math.isfinite(error) and math.isfinite(baseline), "Nonfinite public group-rollout score")
            scores.append({"kind": key[0], "context": key[1], "target": key[2],
                           "all_cached_cells_in_group": len(rows), "effect_mse": error,
                           "zero_effect_mse": baseline})
    means = {kind: float(np.mean([row["effect_mse"] for row in scores if row["kind"] == kind]))
             for kind in ("context", "target")}
    baselines = {kind: float(np.mean([row["zero_effect_mse"] for row in scores if row["kind"] == kind]))
                 for kind in ("context", "target")}
    return {"selection_group_effect_mse": float(np.mean(list(means.values()))),
            "kind_group_effect_mse": means, "kind_zero_effect_mse": baselines,
            "role": "checkpoint_selection_public_development", "checkpoint_selection_affected": True,
            "teacher_forced_interpolants_used": False, "ode_solver": "euler", "ode_steps": 20,
            "flow_start": "zero_effect", "all_cached_group_cells_used_for_means": True,
            "equal_weighting": "groups_within_kind_then_two_kinds", "independent_test": False,
            "improves_zero_effect_each_kind": all(means[k] < baselines[k] for k in means),
            "challenge_treated_used": False, "groups": scores}


def train(args):
    require(args.steps > 0 and args.validate_every > 0 and args.batch_size >= 2 and args.batch_size % 2 == 0,
            "Invalid training length or batch (an even batch is required)")
    require(args.validation_batch_size >= 2 and args.max_validation_groups > 0, "Invalid validation batch")
    require(args.learning_rate > 0 and 0 <= args.dropout < 1, "Invalid optimizer/model configuration")
    require(not args.debug_test_features or args.cpu, "Synthetic features are restricted to CPU tests")
    if not args.cpu:
        require(bool(os.environ.get("SLURM_JOB_ID")), "GPU training requires a Slurm allocation")
        require(torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(0), "An allocated H100 is required")
    device = torch.device("cpu" if args.cpu else "cuda")
    bf16 = device.type == "cuda" and not args.no_bf16
    torch.set_num_threads(max(1, min(int(os.environ.get("SLURM_CPUS_PER_TASK", "4")), 16)))
    require(not args.output_dir.exists() or not any(args.output_dir.iterdir()),
            "Output directory contains artifacts; choose an independent new directory (resume is not implemented)")
    data, cache_sha = load_cache(args.cache)
    require({name.casefold() for name in data["train_contexts"].tolist()} == {"k562"},
            "This controlled variant fits K562 training rows only")
    original_embeddings, embeddings_sha = load_embeddings(args.embeddings)
    if args.debug_test_features:
        embedding_receipt = {"synthetic_test_only": True}
    else:
        require(args.embedding_receipt is not None, "Production training requires an authentic Lingshu receipt")
        embedding_receipt = authenticate_embedding_receipt(args.embedding_receipt, args.embeddings,
                                                            original_embeddings, embeddings_sha)
    requested = set(data["train_targets"].tolist()) | set(data["val_targets"].tolist())
    require(requested <= set(original_embeddings), "Missing Lingshu training/development features")
    embedding_map, mapping_metadata = feature_mapping(original_embeddings, args.feature_mode, args.seed)
    center, scale, transform_metadata = fit_feature_normalization(embedding_map, data["train_targets"])
    transform_metadata.update({
        "mapped_feature_source_genes": [mapping_metadata["gene_feature_mapping"][name]
                                        for name in transform_metadata["training_targets"]],
        "origin_covariates_may_include_other_gene_identities": args.feature_mode == "shuffled",
    })
    source = authenticate_upstream(args.source)
    train_groups, train_effects = group_effects(data, "train")
    require(len(train_groups) >= args.batch_size, "Batch size exceeds distinct training target groups")
    val_groups, val_effects = group_effects(data, "val", validation_groups(data, args.max_validation_groups))
    config = {"gene_count": len(data["gene_names"]), "embedding_dimension": len(center),
              "hidden_size": args.hidden_size, "heads": args.heads, "layers": args.layers,
              "dropout": args.dropout, "feature_mode": args.feature_mode,
              "backbone": "official_scdfm_differential_perceiver", "gene_correlation_graph": False,
              "initialization": "from_scratch", "method": METHOD,
              "context_dependence": "context_agnostic_K562_effect_transferred_onto_recipient_controls",
              "embedding_normalization": "l2_then_train_target_center_scalar_rms",
              "control_anchoring": "pad_inputs_x_t_minus_control_and_zero_control"}
    training_config = {key: getattr(args, key) for key in ("steps", "validate_every", "batch_size",
                       "validation_batch_size", "max_validation_groups", "learning_rate", "seed", "debug_test_features")}
    training_config.update({"bf16": bf16, "objective": "group_effect_conditional_flow_matching_mse_only",
                           "mmd_weight": 0.0, "noise_std": 0.0, "ode_steps": 20,
                           "flow_start": "zero_effect", "training_batch": "distinct_uniform_target_groups",
                           "t0_fraction": 0.5, "remaining_time_distribution": "uniform_0_1",
                           "effect_target": "mean_train_logexpression_minus_mean_train_control_logexpression",
                           "resume_supported": False})
    code_names = ("lingshu_scdfm_effect_model.py", "train_lingshu_scdfm_effect.py",
                  "lingshu_scdfm_model.py", "train_lingshu_scdfm.py")
    code_hashes = {name: sha256_file(Path(__file__).parent / name) for name in code_names}
    selection = "earliest_minimum_equal_kind_mean_all_cell_group_effect_euler20_mse"
    provenance = {"schema": "vcc-lingshu-scdfm-group-effect-training-provenance-v2", "method": METHOD,
                  "upstream": source, "cache_sha256": cache_sha, "embeddings_sha256": embeddings_sha,
                  "embedding_receipt": embedding_receipt, "code_sha256": code_hashes,
                  "model_config": config, "training_config": training_config,
                  "feature_mapping": mapping_metadata, "feature_transform": transform_metadata,
                  "cache_metadata": data["metadata"], "challenge_treated_used": False,
                  "checkpoint_selection": selection, "selection_and_diagnostics_share_public_development_groups": True,
                  "context_dependence": config["context_dependence"],
                  "lingshu_specific_benefit_requires_true_vs_constant_and_shuffled_comparison": True,
                  "independent_test": False, "train_group_count": len(train_groups),
                  "validation_group_count": len(val_groups), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                  "device": str(device), "torch_version": str(torch.__version__),
                  "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output_dir / "training_provenance.json", provenance)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = LingshuScDFMEffect(config, args.source, center, scale).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.learning_rate, weight_decay=1e-4)
    train_features = np.stack([embedding_map[key[-1]] for key, _ in train_groups])
    best_score, best_step = float("inf"), None
    started = time.monotonic()
    for step in range(1, args.steps + 1):
        model.train()
        selected = rng.choice(len(train_groups), args.batch_size, replace=False)
        effect = torch.from_numpy(train_effects[selected]).to(device)
        features = torch.from_numpy(train_features[selected]).to(device)
        t = training_times(args.batch_size, device)
        interpolant = t[:, None] * effect
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=bf16):
            velocity = model(interpolant, torch.zeros_like(effect), features, t)
        loss = torch.nn.functional.mse_loss(velocity.float(), effect)
        require(bool(torch.isfinite(loss)), "Nonfinite training loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        if step == 1 or step % 25 == 0:
            print(json.dumps({"step": step, "group_effect_cfm_mse": float(loss.detach()),
                              "gradient_norm": float(norm), "elapsed_seconds": time.monotonic() - started}), flush=True)
        if step % args.validate_every != 0 and step != args.steps:
            continue
        metrics = validate_group_rollouts(model, val_groups, val_effects, embedding_map, device=device,
                                          batch_size=args.batch_size, bf16=bf16)
        score = metrics["selection_group_effect_mse"]
        improved = score < best_score
        if improved:
            best_score, best_step = score, step
        payload = {"schema": CHECKPOINT_SCHEMA, "method": METHOD, "step": step,
                   "best_step": best_step, "best_score": best_score, "validation": metrics,
                   "model_config": config, "training_config": training_config, "upstream": source,
                   "cache_sha256": cache_sha, "embeddings_sha256": embeddings_sha,
                   "embedding_receipt": embedding_receipt, "code_sha256": code_hashes,
                   "feature_mapping": mapping_metadata, "feature_transform": transform_metadata,
                   "gene_names": data["gene_names"].tolist(), "gene_indices": data["gene_indices"].tolist(),
                   "normalization_gene_names": data["normalization_gene_names"].tolist(),
                   "metadata": data["metadata"], "model_state": model.state_dict(),
                   "optimizer_state": optimizer.state_dict(), "torch_rng_state": torch.get_rng_state(),
                   "cuda_rng_states": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                   "numpy_rng_state": rng.bit_generator.state, "challenge_treated_used": False}
        archive = args.output_dir / f"step_{step:07d}.pt"
        require(not archive.exists(), "Refusing to overwrite checkpoint archive")
        save_checkpoint(archive, payload)
        save_checkpoint(args.output_dir / "last.pt", payload)
        if improved:
            save_checkpoint(args.output_dir / "best.pt", payload)
        atomic_json(args.output_dir / f"validation_{step:07d}.json", metrics)
        print(json.dumps({"step": step, "selection_group_effect_mse": score, "best_step": best_step}), flush=True)
    selected_checkpoint = torch.load(args.output_dir / "best.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(selected_checkpoint["model_state"], strict=True)
    diagnostics = rollout_diagnostics(model, data, embedding_map, val_groups, device=device,
                                      batch_size=args.validation_batch_size, noise_std=0.0,
                                      seed=args.seed + 1771, bf16=bf16)
    diagnostics.update({"method": METHOD, "noise_std": 0.0, "control_anchoring": config["control_anchoring"],
                        "selection_used_same_public_development_groups": True, "independent_test": False,
                        "checkpoint_selection_metric": selection,
                        "selection_metric_differs_from_sampled_cell_diagnostic": True,
                        "selection_used_all_cell_group_mean_effects": True})
    atomic_json(args.output_dir / "best_rollout_diagnostics.json", diagnostics)
    summary = {"schema": "vcc-lingshu-scdfm-training-summary-v1", "method": METHOD,
               "checkpoint_schema": CHECKPOINT_SCHEMA, "status": "completed", "feature_mode": args.feature_mode,
               "last_step": args.steps, "best_step": best_step, "best_score": best_score,
               "best_checkpoint_sha256": sha256_file(args.output_dir / "best.pt"),
               "training_provenance_sha256": sha256_file(args.output_dir / "training_provenance.json"),
               "elapsed_seconds": time.monotonic() - started, "challenge_treated_used": False,
               "control_anchoring": config["control_anchoring"], "checkpoint_selection": selection,
               "context_dependence": config["context_dependence"],
               "selection_and_diagnostics_share_public_development_groups": True, "independent_test": False,
               "public_model_quality_evaluated": True, "official_challenge_metrics_evaluated": False,
               "validation_metric_scope": "public_development_group_effect_rollout_selection_then_sampled_cell_diagnostics",
               "deployment_rollout_diagnostics": diagnostics}
    atomic_json(args.output_dir / "training_summary.json", summary)
    return summary


if __name__ == "__main__":
    print(json.dumps(train(parser().parse_args()), indent=2))
