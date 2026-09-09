#!/usr/bin/env python3
"""Credential-free preupload checks; empirical safeguards, not score guarantees."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = "vcc-lingshu-scdfm-preupload-audit-v1"
MODEL_ID = "lingshu-medical-mllm/Lingshu-7B"
REVISION = "b98aecd41dfd9d7545a6b8e2f4743ae8471bd7a9"
MAX_HIGH_CLAMP = 0.01
LIBRARY_RATIO_BOUNDS = (0.5, 2.0)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f"Nonfinite JSON value: {value}")
    return json.loads(path.read_text(), parse_constant=reject)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value), "Metric must be a finite number")
    return value


def audit(run_dir):
    result = {"schema": SCHEMA, "status": "fail", "errors": [], "hashes": {},
              "safeguards": {"max_high_clamp": MAX_HIGH_CLAMP, "library_ratio_bounds": list(LIBRARY_RATIO_BOUNDS)},
              "interpretation": "Empirical preupload safeguards; not proof of improved challenge scores",
              "audit_script_sha256": digest(Path(__file__))}
    try:
        names = {"summary": "fit/training_summary.json", "provenance": "fit/training_provenance.json",
                 "checkpoint": "fit/best.pt", "rollout": "fit/best_rollout_diagnostics.json",
                 "generation": "prediction.h5ad.receipt.json", "prediction": "prediction.h5ad",
                 "package": "prediction.vcc", "embedding_receipt": "lingshu_embeddings.json",
                 "embeddings": "lingshu_embeddings.npz", "prep_validation": "prep_validation.json",
                 "prep_package": "prep_package.json"}
        paths = {key: run_dir / value for key, value in names.items()}
        hashes = {key: digest(path) for key, path in paths.items()}
        result["hashes"] = hashes
        s, p, g = (read_json(paths[key]) for key in ("summary", "provenance", "generation"))
        for obj, schema in ((s, "training-summary"), (p, "training-provenance"), (g, "generation-receipt")):
            require(obj.get("schema") == f"vcc-lingshu-scdfm-{schema}-v1", f"Wrong {schema} schema")
            require(obj.get("challenge_treated_used") is False, "Challenge treated input is forbidden")
        require(s.get("status") == g.get("status") == "completed", "Training or generation incomplete")
        require(s.get("public_model_quality_evaluated") is True and s.get("official_challenge_metrics_evaluated") is False,
                "Expected completed public diagnostics before first official submission")
        require(g.get("test_subset") is False and g.get("manual_target_effects") is False, "Test/manual output forbidden")
        require(p["training_config"].get("debug_test_features") is False, "Synthetic feature training forbidden")
        require(p["model_config"]["gene_count"] == g["modeled_gene_count"] == 512, "Expected 512 modeled genes")
        require(s["best_checkpoint_sha256"] == g["checkpoint_sha256"] == hashes["checkpoint"], "Checkpoint hash mismatch")
        require(g["training_summary_sha256"] == hashes["summary"], "Summary hash mismatch")
        require(s["training_provenance_sha256"] == hashes["provenance"], "Training provenance hash mismatch")
        require(g["prediction_sha256"] == hashes["prediction"], "Prediction hash mismatch")
        require(p["embeddings_sha256"] == g["embeddings_sha256"] == hashes["embeddings"], "Embedding hash mismatch")
        require(p["embedding_receipt_sha256"] == p["embedding_receipt"]["sha256"] == hashes["embedding_receipt"], "Embedding receipt mismatch")
        e = read_json(paths["embedding_receipt"])
        require(e.get("status") == "complete" and e.get("model_id") == MODEL_ID and e.get("model_revision") == REVISION,
                "Expected genuine completed pinned Lingshu extraction")
        require(e.get("embedding_dimension") == 3584 and e.get("challenge_treated_data_used") is False, "Wrong embedding scope")
        require(e["npz"]["sha256"] == hashes["embeddings"], "Extraction NPZ mismatch")
        require(g["shape"] == [360000, 18533] and g["normalization_gene_count"] == 18077, "Wrong official dimensions")
        require(0 < finite(g["nnz"]) <= 4750000000, "Prediction density exceeds official limit")
        require(finite(s["best_score"]) >= 0 and 0 < s["best_step"] <= s["last_step"], "Invalid selected training step/score")
        rollout = read_json(paths["rollout"])
        require(s["deployment_rollout_diagnostics"] == rollout, "Rollout diagnostics disagree with summary")
        require(rollout.get("checkpoint_selection_affected") is False and rollout.get("challenge_treated_used") is False,
                "Rollout must be public-only and after frozen selection")
        require(rollout.get("ode_steps") == 20 and set(rollout["kind_metrics"]) == {"context", "target"}, "Both rollout kinds required")
        for kind, metrics in rollout["kind_metrics"].items():
            require(finite(metrics["model_centroid_mse"]) >= 0 and finite(metrics["control_centroid_mse"]) >= 0, "Negative centroid MSE")
            for metric in ("centroid_mse", "mmd2"):
                require(finite(metrics[f"model_{metric}"]) < finite(metrics[f"control_{metric}"]), f"{kind} {metric} does not improve controls")
            require(0 <= finite(metrics["clamped_high_fraction"]) <= MAX_HIGH_CLAMP, f"{kind} high clamp exceeds 1%")
            require(0 <= finite(metrics["clamped_low_fraction"]) <= 1, "Invalid rollout low-clamp fraction")
        clamps = g["clamping"]
        require(clamps["modeled_values"] == 360000 * 512, "Wrong modeled value count")
        high = finite(clamps["clamped_high"]) / clamps["modeled_values"]
        low = finite(clamps["clamped_low"]) / clamps["modeled_values"]
        require(0 <= high <= MAX_HIGH_CLAMP and 0 <= low <= 1, "Generation clamp limits failed")
        groups = g["group_receipts"]
        require(len(groups) == 900, "Expected exactly 900 context/target groups")
        seen, targets_by_context, ratios = set(), {context: set() for context in "ABC"}, []
        for group in groups:
            context, target = group["context"], group["target_gene"]
            require(context in targets_by_context and isinstance(target, str) and target and target != "non-targeting", "Invalid group")
            require((context, target) not in seen, "Duplicate context/target group")
            seen.add((context, target)); targets_by_context[context].add(target)
            ratio = finite(group["library_sum_ratio"])
            require(LIBRARY_RATIO_BOUNDS[0] <= ratio <= LIBRARY_RATIO_BOUNDS[1], "Group library ratio outside [0.5,2.0]")
            before, after = finite(group["input_library_sum"]), finite(group["output_library_sum"])
            require(before > 0 and math.isclose(ratio, after / before, rel_tol=1e-12), "Inconsistent library ratio")
            ratios.append(ratio)
        require(len(targets_by_context["A"]) == 300 and targets_by_context["A"] == targets_by_context["B"] == targets_by_context["C"], "Context target panels differ")
        for key, dry in (("prep_validation", True), ("prep_package", False)):
            prep = read_json(paths[key])
            require(prep.get("dry_run") is dry and prep.get("verified_targets") is True, "Official prep not successful/target-verified")
            require((prep["n_cells"], prep["n_genes"]) == (360000, 18533) and prep["normalization"] == "counts-preserved", "Official prep shape/count mode mismatch")
            require(prep["cells_per_context"] == {context: 120000 for context in "ABC"} and prep["nnz"] == g["nnz"], "Official prep context/density mismatch")
            require(Path(prep["input"]).resolve() == paths["prediction"].resolve(), "Official prep input path mismatch")
            if not dry:
                require(Path(prep["output"]).resolve() == paths["package"].resolve(), "Official prep output path mismatch")
        checksum = (run_dir / "prediction.vcc.sha256").read_text().strip().split(maxsplit=1)
        require(len(checksum) == 2 and checksum[0] == hashes["package"] and Path(checksum[1].lstrip("*")).name == "prediction.vcc", "Package SHA256 receipt mismatch")
        result.update(status="pass", rollout=rollout["kind_metrics"], generation_high_clamp_fraction=high,
                      generation_low_clamp_fraction=low, library_ratio_range=[min(ratios), max(ratios)])
    except (OSError, ValueError, KeyError, TypeError, IndexError, ZeroDivisionError) as error:
        result["errors"].append(str(error))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/lingshu_scdfm/v1"))
    args = parser.parse_args()
    result = audit(args.run_dir)
    temporary = args.run_dir / f"audit.json.tmp.{os.getpid()}"
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, args.run_dir / "audit.json")
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
