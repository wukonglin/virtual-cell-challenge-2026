"""Frozen public-flow diagnosis: register inputs, then run by contract SHA.

No fitting, raw expression files, credentials, downloads or model selection.
Registration hashes opaque NPZs. Execution reads only the authenticated cache,
registered GO features and safe NPZ final weights, on a BioHPC compute host.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import numpy as np
import torch
import train_public_flow_pilot as pilot
from public_crispri_flow import ConditionalPublicFlow, FlowConfig, sample_heun

SCHEMA = "public-flow-posthoc-diagnostic-contract-v1"
ARMS = ("true", "constant", "shuffled")
INPUTS = {"cache": "cache.npz", "contract": "contract.json", "source-manifest": "sources.json",
          "target-features": "go/features.npz", "feature-receipt": "go/receipt.json"}
MODULES = ("diagnose_public_flow_pilot.py", "train_public_flow_pilot.py", "public_crispri_flow.py",
           "public_flow_conditioning.py", "build_go_target_features.py")
METHOD = {
    "scope": "post-hoc development diagnosis; no training, selection, promotion or submission",
    "splits": ["train", "dev"], "arms": list(ARMS), "device": "cpu", "torch_threads": 2,
    "seed": 20260910, "heun_steps": 16, "maximum_group_cells": 32, "clipping": False,
    "replay": "original dev RNG and final weights; require stored group metrics within tolerance",
    "replay_rtol": 0.0002, "replay_atol": 0.000002,
    "observed_shift": "treated mean minus sampled control mean",
    "predicted_shift": "predicted mean minus identical starting control mean",
    "displacement": "Euclidean centroid displacement; unpaired, not measured cell trajectories",
    "alignment": "cosine of predicted and observed shifts; null when norm product <= 1e-12",
    "variance": "population gene variance averaged over genes; compare to controls and treated separately",
    "shared_shift": "within source, mean batches per target, equal target weights; squared norm of mean shift divided by mean squared shift norm",
    "sham": "all-zero target vector INCLUDING coverage bits; OOD, not a trained NTC condition",
    "sham_sampling": "disjoint NTC halves per group; RNG seed 20260911; max32 per half; context remains full registered control mean/std",
    "identity": "zero-initialized velocity head and 16-step Heun preserve controls exactly",
    "independence": "groups can reuse controls; group metrics are not independent biological replicates",
    "context_shift": "raw matched-control context distance to training contexts; no fitted threshold",
}


def _entry(path):
    return {"path": str(path.absolute()), "sha256": pilot.sha256_file(path), "size_bytes": path.stat().st_size}


def _check_entry(entry):
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256", "size_bytes"}:
        raise ValueError("Malformed authenticated file entry")
    path = Path(entry["path"])
    if not path.is_absolute() or type(entry["size_bytes"]) is not int or entry["size_bytes"] < 1:
        raise ValueError("Absolute path and positive file size required")
    pilot.authenticated(path, entry["sha256"])
    if path.stat().st_size != entry["size_bytes"]:
        raise ValueError("Registered file size changed")
    return path


def verify_completion(entries, expected_job):
    """Authenticate suite/summaries/tracking before any array decoding."""
    paths = {key: _check_entry(entry) for key, entry in entries.items()}
    read = lambda key: pilot.authenticated_json(paths[key], entries[key]["sha256"])
    suite = read("suite")
    if (suite.get("schema") != "public-flow-pilot-suite-v1" or suite.get("completed") is not True
            or suite.get("stage") != "pretrain" or suite.get("slurm_job_id") != expected_job
            or suite.get("promotion_performed") is not False or suite.get("submission_performed") is not False
            or suite.get("tracking_mode") != "offline" or suite.get("cloud_synced") is not False):
        raise ValueError("Not the specified completed unpromoted offline suite")
    expected_inputs = {key: entries[key]["sha256"] for key in INPUTS}
    if (suite.get("input_sha256") != expected_inputs or suite.get("contract_sha256") != expected_inputs["contract"]
            or suite.get("cache_sha256") != expected_inputs["cache"]):
        raise ValueError("Suite input identities differ")
    outcomes = suite.get("arms")
    if (not isinstance(outcomes, list) or len(outcomes) != 3 or any(not isinstance(x, dict) for x in outcomes)
            or [x.get("arm") for x in outcomes] != list(ARMS)):
        raise ValueError("Exactly three registered completed arms required")
    summaries, run_ids = {}, set()
    for mode, outcome in zip(ARMS, outcomes):
        summary, tracking = read(f"{mode}/summary"), read(f"{mode}/tracking")
        if (outcome.get("exit_code") != 0 or outcome.get("summary_sha256") != entries[f"{mode}/summary"]["sha256"]
                or outcome.get("tracking_receipt_sha256") != entries[f"{mode}/tracking"]["sha256"]):
            raise ValueError("Suite arm receipt identities differ")
        if set(summary.get("configuration", {})) != set(asdict(pilot.PilotConfig())):
            raise ValueError("Missing complete checkpoint configuration")
        cfg = pilot.PilotConfig(**summary["configuration"])
        if cfg != pilot.PilotConfig(feature_mode=mode):
            raise ValueError("Unexpected completed pilot hyperparameters")
        if (summary.get("schema") != "vcc-public-flow-pilot-summary-v1" or summary.get("status") != "completed_unpromoted_pilot"
                or summary.get("stage") != "pretrain" or summary.get("last_step") != 200
                or summary.get("weights_sha256") != entries[f"{mode}/weights"]["sha256"]
                or summary.get("contract_sha256") != expected_inputs["contract"]
                or summary.get("source_manifest_sha256") != expected_inputs["source-manifest"]
                or summary.get("representation") != pilot.REPRESENTATION
                or summary.get("runtime", {}).get("slurm_job_id") != expected_job
                or summary.get("dev_used_for_fitting_or_selection") is not False
                or summary.get("checkpoint_selection") != "fixed_final_step_only"):
            raise ValueError("Checkpoint identity or no-fitting status differs")
        provenance = summary.get("provenance", {})
        mapping = {"cache_sha256": "cache", "target_features_sha256": "target-features", "feature_receipt_sha256": "feature-receipt",
                   "entrypoint_sha256": "module/train_public_flow_pilot.py", "flow_module_sha256": "module/public_crispri_flow.py",
                   "conditioning_module_sha256": "module/public_flow_conditioning.py", "go_builder_sha256": "module/build_go_target_features.py"}
        if any(provenance.get(key) != entries[value]["sha256"] for key, value in mapping.items()):
            raise ValueError("Checkpoint provenance or implementation bytes changed")
        if (tracking.get("execution_status") != "completed" or tracking.get("training_exit_code") != 0
                or tracking.get("tracking_errors") != 0 or tracking.get("mode") != "offline"
                or tracking.get("cloud_synced") is not False or tracking.get("event_count", 0) < 200
                or not isinstance(tracking.get("run_id"), str)
                or tracking.get("run_id") != outcome.get("run_id") or tracking.get("run_id") in run_ids):
            raise ValueError("Incomplete or reused W&B recording")
        run_ids.add(tracking["run_id"])
        summaries[mode] = summary
    return paths, summaries


def register(pilot_dir, completed_dir, output, *, expected_job="20539599"):
    if os.path.lexists(output):
        raise FileExistsError("Diagnostic registration must be fresh")
    paths = {key: pilot_dir / name for key, name in INPUTS.items()}
    paths["suite"] = completed_dir / "suite.json"
    for mode in ARMS:
        paths[f"{mode}/summary"] = completed_dir / "runs" / mode / "summary.json"
        paths[f"{mode}/weights"] = completed_dir / "runs" / mode / "weights.npz"
        paths[f"{mode}/tracking"] = completed_dir / "tracking" / mode / "tracking.json"
    paths.update({"module/" + name: Path(__file__).with_name(name) for name in MODULES})
    entries = {key: _entry(path) for key, path in paths.items()}
    verify_completion(entries, expected_job)
    registration = {"schema": SCHEMA, "created_at": datetime.now(timezone.utc).isoformat(),
                    "expected_job": expected_job, "method": METHOD, "inputs": entries,
                    "registration_npz_decoded": False, "raw_source_files_opened": False}
    pilot.write_new(output, (json.dumps(registration, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return pilot.sha256_file(output)


def restore_model(cache, table, summary, weights_path, weights_sha256):
    cfg = pilot.PilotConfig(**summary["configuration"])
    conditions, transform, coverage = pilot.target_conditions(cache, table, cfg)
    expected_cfg = FlowConfig(len(cache.genes), conditions["train"].shape[1], len(cache.genes) * 2, cfg.hidden_size, cfg.layers)
    if summary.get("model_config") != asdict(expected_cfg) or summary.get("output_gene_count") != len(cache.genes):
        raise ValueError("Checkpoint architecture differs from authenticated inputs")
    canonical_transform = json.loads(json.dumps(transform.provenance(), allow_nan=False))
    if summary.get("target_transform") != canonical_transform or summary.get("go_targets_present") != coverage:
        raise ValueError("Checkpoint transform provenance differs from training-only reconstruction")
    model = ConditionalPublicFlow(expected_cfg)
    state = model.state_dict()
    arrays = pilot.load_npz(weights_path, weights_sha256, {"model." + key for key in state} | {"target_mean", "target_scale"})
    for name, expected in (("target_mean", transform.mean), ("target_scale", transform.scale)):
        value = arrays[name]
        if value.dtype != expected.dtype or value.shape != expected.shape or not np.array_equal(value, expected):
            raise ValueError("Stored GO mean/scale differs from exact training-only reconstruction")
    restored = {}
    for name, expected in state.items():
        value = arrays["model." + name]
        if value.dtype != expected.numpy().dtype or value.shape != tuple(expected.shape) or not np.isfinite(value).all():
            raise ValueError("Weight dtype, shape or finiteness mismatch")
        restored[name] = torch.from_numpy(value.copy())
    model.load_state_dict(restored, strict=True)
    model.eval()
    return model, conditions


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator > 1e-12 else None


def _cosine(left, right):
    scale = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.clip(np.dot(left, right) / scale, -1, 1)) if scale > 1e-12 else None


def displacement_metrics(predicted, control, treated):
    predicted, control, treated = (np.asarray(x, dtype=np.float64) for x in (predicted, control, treated))
    if (any(x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all() for x in (predicted, control, treated))
            or predicted.shape != control.shape or treated.shape[1] != control.shape[1]):
        raise ValueError("Finite compatible cells-by-genes populations required")
    dp, dt = predicted.mean(0) - control.mean(0), treated.mean(0) - control.mean(0)
    pn, tn = float(np.linalg.norm(dp)), float(np.linalg.norm(dt))
    pv, cv, tv = (float(x.var(0).mean()) for x in (predicted, control, treated))
    result = pilot.diagnostics(predicted, control, treated)
    result.update(predicted_displacement_l2=pn, observed_displacement_l2=tn,
                  displacement_norm_ratio=_ratio(pn, tn), displacement_cosine=_cosine(dp, dt),
                  per_cell_displacement_rms=float(np.sqrt(np.square(predicted - control).mean())),
                  centroid_shift_mse=float(np.square(dp).mean()),
                  predicted_shift_rms=float(np.sqrt(np.square(dp).mean())),
                  observed_shift_rms=float(np.sqrt(np.square(dt).mean())),
                  shift_dot_mean=float(np.dot(dp, dt) / len(dp)), shift_cosine=_cosine(dp, dt),
                  predicted_variance=pv, control_variance=cv, treated_variance=tv,
                  variance_ratio_to_control=_ratio(pv, cv), variance_ratio_to_treated=_ratio(pv, tv),
                  minimum_prediction=float(predicted.min()), mean_negative_magnitude=float(np.maximum(-predicted, 0).mean()),
                  negative_gene_fraction=float(np.any(predicted < 0, axis=0).mean()),
                  mse_excess_decomposition=float((np.dot(dp, dp) - 2 * np.dot(dp, dt)) / len(dp)))
    if not np.isclose(result["mse_excess_decomposition"], result["model_centroid_mse"] - result["control_centroid_mse"], atol=1e-10, rtol=1e-9):
        raise AssertionError("Centroid error decomposition failed")
    return result, dp, dt


def shared_shift(rows, predicted, observed):
    result = {}
    for source in sorted({row["source"] for row in rows}):
        targets = sorted({row["target"] for row in rows if row["source"] == source})
        axes = []
        for vectors in (predicted, observed):
            per_target = np.stack([np.mean([vector for row, vector in zip(rows, vectors)
                                           if (row["source"], row["target"]) == (source, target)], axis=0) for target in targets])
            common = per_target.mean(0)
            energy = np.square(per_target).sum(1).mean()
            cosines = [_cosine(per_target[i], per_target[j]) for i in range(len(targets)) for j in range(i)]
            defined = [value for value in cosines if value is not None]
            axes.append({"target_count": len(targets), "common_displacement_l2": float(np.linalg.norm(common)),
                         "common_shift_energy_fraction": _ratio(float(np.dot(common, common)), float(energy)),
                         "mean_pairwise_target_cosine": float(np.mean(defined)) if defined else None})
        result[source] = {"predicted": axes[0], "observed": axes[1]}
    return result


def balanced_aggregate(rows):
    result = {}
    for key in sorted(set(rows[0]) - {"group", "source", "target", "cells"}):
        sources = []
        for source in sorted({row["source"] for row in rows}):
            targets = []
            for target in sorted({row["target"] for row in rows if row["source"] == source}):
                values = [row[key] for row in rows if (row["source"], row["target"]) == (source, target) and row[key] is not None]
                if values:
                    targets.append(float(np.mean(values)))
            if targets:
                sources.append(float(np.mean(targets)))
        result[key] = float(np.mean(sources)) if sources else None
    return result


def _rollout(model, control, target, context):
    x = torch.from_numpy(control)
    return sample_heun(model, x, torch.from_numpy(np.repeat(target[None], len(x), axis=0)),
                       torch.from_numpy(np.repeat(context[None], len(x), axis=0)),
                       torch.ones_like(x, dtype=torch.bool), steps=METHOD["heun_steps"]).cpu().numpy()


def diagnose_split(model, split, conditions, *, sham=False):
    rng = np.random.default_rng(METHOD["seed"] + int(sham))
    rows, shifts, observed = [], [], []
    for group, (source, target) in enumerate(zip(split.sources, split.targets)):
        ci, ti = np.flatnonzero(split.control_group == group), np.flatnonzero(split.group == group)
        if sham:
            n = min(32, len(ci) // 2)
            permuted = rng.permutation(ci)
            control, treated = split.control[permuted[:n]], split.control[permuted[n:2*n]]
            features = np.zeros_like(conditions[group])
        else:
            n = min(32, len(ci), len(ti))
            control = split.control[rng.choice(ci, n, replace=False)]
            treated = split.treated[rng.choice(ti, n, replace=False)]
            features = conditions[group]
        predicted = _rollout(model, control, features, split.context[group])
        metrics, dp, dt = displacement_metrics(predicted, control, treated)
        rows.append({"group": group, "source": source, "target": target, "cells": n, **metrics})
        shifts.append(dp)
        observed.append(dt)
    return {"groups": rows, "aggregate": balanced_aggregate(rows), "shared_shift_by_source": shared_shift(rows, shifts, observed)}


def replay_check(actual, summary):
    expected = summary.get("dev_groups")
    if not isinstance(expected, list) or len(actual) != len(expected):
        raise ValueError("Development replay group count differs")
    maximum = 0.0
    for got, wanted in zip(actual, expected):
        for key, value in wanted.items():
            if key in {"group", "source", "target", "cells"} or value is None:
                if got.get(key) != value:
                    raise ValueError("Development replay group identity/nullability differs")
            elif not np.isclose(got[key], value, rtol=METHOD["replay_rtol"], atol=METHOD["replay_atol"]):
                raise ValueError("Frozen development diagnostics did not reproduce")
            else:
                maximum = max(maximum, abs(float(got[key]) - value))
    return {"passed": True, "maximum_absolute_metric_difference": maximum,
            "rtol": METHOD["replay_rtol"], "atol": METHOD["replay_atol"]}


def identity_check(cache, target_dim):
    zero_velocity = ConditionalPublicFlow(FlowConfig(len(cache.genes), target_dim, len(cache.genes) * 2))
    maximum = 0.0
    for split in (cache.train, cache.dev):
        control = split.control[:32]
        predicted = _rollout(zero_velocity, control, np.zeros(target_dim, dtype=np.float32), split.context[0])
        if not np.array_equal(predicted, control):
            raise AssertionError("Zero-velocity Heun did not preserve controls exactly")
        maximum = max(maximum, float(np.abs(predicted - control).max()))
    return {"passed": True, "maximum_absolute_coordinate_difference": maximum,
            "meaning": "analytic implementation check, not trained NTC-null conditioning"}


def context_shift(cache):
    rows = []
    for group, context in enumerate(cache.dev.context):
        distances = np.square(cache.train.context.astype(np.float64) - context.astype(np.float64)).mean(1)
        rows.append({"group": group, "source": cache.dev.sources[group], "target": cache.dev.targets[group],
                     "nearest_training_context_rms": float(np.sqrt(distances.min())),
                     "mean_training_context_rms": float(np.sqrt(distances).mean()),
                     "control_mean_expression": float(context[:len(cache.genes)].mean())})
    return rows


def run(registration_path, registration_sha256, output):
    if os.path.lexists(output):
        raise FileExistsError("Diagnostic results must be fresh; no silent rerun")
    registration = pilot.authenticated_json(registration_path, registration_sha256)
    if (registration.get("schema") != SCHEMA or registration.get("method") != METHOD
            or registration.get("registration_npz_decoded") is not False or registration.get("raw_source_files_opened") is not False):
        raise ValueError("Diagnostic methods differ from frozen registration")
    entries = registration["inputs"]
    expected_keys = set(INPUTS) | {"suite"} | {f"{arm}/{key}" for arm in ARMS for key in ("summary", "weights", "tracking")} | {"module/" + name for name in MODULES}
    if set(entries) != expected_keys:
        raise ValueError("Diagnostic input roster differs")
    for name in MODULES:
        if entries["module/" + name]["sha256"] != pilot.sha256_file(Path(__file__).with_name(name)):
            raise ValueError("Executing code changed since diagnostic registration")
    paths, summaries = verify_completion(entries, registration["expected_job"])
    sha = lambda key: entries[key]["sha256"]
    cache = pilot.load_cache(paths["cache"], sha("cache"), sha("contract"), sha("source-manifest"))
    pilot.validate_registration(cache, pilot.PilotConfig(), paths["contract"], paths["source-manifest"])
    table = pilot.load_go_table(paths["target-features"], sha("target-features"), paths["feature-receipt"], sha("feature-receipt"), cache.train.targets)
    arms = {}
    for mode in ARMS:
        model, conditions = restore_model(cache, table, summaries[mode], paths[f"{mode}/weights"], sha(f"{mode}/weights"))
        diagnostic = {split: diagnose_split(model, getattr(cache, split), conditions[split]) for split in ("train", "dev")}
        diagnostic["original_dev_replay"] = replay_check(diagnostic["dev"]["groups"], summaries[mode])
        diagnostic["ood_zero_target_sham"] = {"not_a_trained_ntc_condition": True, "all_target_features_including_masks_zeroed": True,
                                              **{split: diagnose_split(model, getattr(cache, split), conditions[split], sham=True) for split in ("train", "dev")}}
        arms[mode] = diagnostic
    tracking_keys = {"model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2", "variance_ratio",
                     "covariance_effective_rank", "zero_fraction", "duplicate_fraction", "negative_fraction",
                     "predicted_shift_rms", "observed_shift_rms", "shift_dot_mean", "shift_cosine"}
    rollout = {"kind_metrics": {"context": {key: value for key, value in arms["true"]["dev"]["aggregate"].items()
                                             if key in tracking_keys and value is not None}}}
    result = {"schema": "public-flow-posthoc-diagnostic-results-v1", "diagnostic_contract_sha256": registration_sha256,
              "slurm_training_job": registration["expected_job"], "method": METHOD, "arms": arms,
              "zero_velocity_identity": identity_check(cache, conditions["train"].shape[1]),
              "context_shift": context_shift(cache), "new_fitting_performed": False, "checkpoint_selection_performed": False,
              "raw_source_files_opened": False, "submission_performed": False,
              "tracking_rollout_arm": "true", "deployment_rollout_diagnostics": rollout,
              "feature_transform": "training-target-only reconstruction, exact saved means/scales verified",
              "runtime": {"device": "cpu", "torch_threads": torch.get_num_threads(), "hostname": socket.gethostname()}}
    pilot.write_new(output, (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("register")
    prep.add_argument("--pilot-dir", type=Path, required=True)
    prep.add_argument("--completed-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--expected-job", default="20539599")
    execute = commands.add_parser("run")
    execute.add_argument("--diagnostic-contract", type=Path, required=True)
    execute.add_argument("--diagnostic-contract-sha256", required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "register":
        print(json.dumps({"diagnostic_contract_sha256": register(args.pilot_dir, args.completed_dir, args.output, expected_job=args.expected_job),
                          "numeric_arrays_decoded": False, "next": "review frozen registration before run"}))
    else:
        if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
            raise RuntimeError("Run CPU diagnostics only on approved BioHPC compute hosts, never an HPC head node")
        torch.set_num_threads(2)
        result = run(args.diagnostic_contract, args.diagnostic_contract_sha256, args.output)
        print(json.dumps({"step": 200, "deployment_rollout_diagnostics": result["deployment_rollout_diagnostics"]}, allow_nan=False))
        print(json.dumps({"result_sha256": pilot.sha256_file(args.output), "fitting_performed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
