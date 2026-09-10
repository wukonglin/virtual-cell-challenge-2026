#!/usr/bin/env python3
"""Report comparable public-development ablations; never select a checkpoint."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

try:
    from .check_lingshu_scdfm_public_quality import check_public_quality
except ImportError:
    from check_lingshu_scdfm_public_quality import check_public_quality

ARMS = ("true", "shuffled", "constant")
METRICS = ("centroid_mse", "mmd2")
METHOD = "control_anchored_public_group_effect_cfm_v2"


def compare(root):
    root = Path(root)
    result = {"schema": "vcc-lingshu-scdfm-effect-comparison-v2", "status": "fail", "errors": [],
              "arms": {}, "true_minus_comparator": {}, "lingshu_benefit_claim": "not established",
              "interpretation": "development comparison only, not independent test",
              "checkpoint_selection_affected": False, "independent_test": False,
              "challenge_treated_used": False}
    comparable = {}

    def require(ok, message):
        if not ok:
            raise ValueError(message)

    def digest(value, size=64):
        return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{size}}}", value) is not None

    def read(path):
        payload = path.read_bytes()

        def reject(value):
            raise ValueError(f"Nonfinite JSON value: {value}")

        obj = json.loads(payload, parse_constant=reject)
        require(isinstance(obj, dict), f"{path.name} must be an object")
        return obj, hashlib.sha256(payload).hexdigest()

    for arm in ARMS:
        fit = root / arm / "fit"
        quality = check_public_quality(fit / "training_summary.json")
        report = {"public_quality_status": quality["status"], "public_quality_errors": quality["errors"],
                  "public_metrics": quality["kind_metrics"], "metric_ratios": quality["metric_ratios"],
                  "training_summary_sha256": quality["training_summary_sha256"], "best_step": None}
        result["arms"][arm] = report
        try:
            summary, summary_sha = read(fit / "training_summary.json")
            provenance, provenance_sha = read(fit / "training_provenance.json")
            report.update(best_step=summary.get("best_step"), training_provenance_sha256=provenance_sha)
            require(summary_sha == quality["training_summary_sha256"], "Summary changed during comparison")
            require(summary.get("schema") == "vcc-lingshu-scdfm-training-summary-v1" and
                    summary.get("status") == "completed", "Completed training summary required")
            require(provenance.get("schema") == "vcc-lingshu-scdfm-group-effect-training-provenance-v2", "Wrong provenance schema")
            require(summary.get("training_provenance_sha256") == provenance_sha, "Provenance digest mismatch")
            require(summary.get("method") == provenance.get("method") == METHOD, "Wrong effect-v2 method")
            for obj in (summary, provenance):
                require(obj.get("challenge_treated_used") is False, "Public-only data required")
                require(obj.get("independent_test") is False and
                        obj.get("selection_and_diagnostics_share_public_development_groups") is True,
                        "Shared public-development scope must be acknowledged")
            require(summary.get("public_model_quality_evaluated") is True and
                    summary.get("official_challenge_metrics_evaluated") is False, "Public diagnostics only required")
            model, training, mapping = (provenance[key] for key in ("model_config", "training_config", "feature_mapping"))
            require(all(isinstance(obj, dict) for obj in (model, training, mapping)), "Malformed arm configuration")
            require(summary.get("feature_mode") == model.get("feature_mode") == mapping.get("mode") == arm,
                    "Feature-mode labels do not match arm directory")
            require("feature_mode" not in training or training["feature_mode"] == arm, "Training feature-mode label mismatch")
            require(training.get("debug_test_features") is False, "Synthetic test features forbidden")
            require(mapping.get("challenge_expression_or_labels_used") is False, "Feature mapping must not use challenge data")
            best, last = summary.get("best_step"), summary.get("last_step")
            require(type(best) is int and type(last) is int and 0 < best <= last == training.get("steps"), "Invalid training completion step")
            score = summary.get("best_score")
            require(type(score) in (int, float) and math.isfinite(score) and score >= 0, "Invalid selected score")
            for key in ("cache_sha256", "embeddings_sha256"):
                require(digest(provenance.get(key)), f"Missing or malformed {key}")
            code, upstream = provenance.get("code_sha256"), provenance.get("upstream")
            require(isinstance(code, dict) and bool(code) and all(digest(value) for value in code.values()), "Invalid code digests")
            require(isinstance(upstream, dict) and all(digest(upstream.get(key), 40) for key in ("commit", "tree")), "Invalid upstream identity")
            receipt = provenance.get("embedding_receipt")
            require(isinstance(receipt, dict) and digest(receipt.get("sha256")), "Authentic embedding receipt digest required")
            require(receipt.get("npz_sha256") == provenance["embeddings_sha256"], "Embedding receipt NPZ digest mismatch")
            rollout = summary["deployment_rollout_diagnostics"]
            require(rollout.get("role") == "diagnostic_only_after_checkpoint_selection" and
                    rollout.get("checkpoint_selection_affected") is False and
                    rollout.get("challenge_treated_used") is False and
                    rollout.get("official_challenge_metrics_evaluated") is False and
                    rollout.get("independent_test") is False and
                    rollout.get("selection_used_same_public_development_groups") is True,
                    "Final diagnostics must be public-development only and after checkpoint selection")
            groups = rollout.get("groups")
            require(isinstance(groups, list) and bool(groups), "Final diagnostic groups required")
            keys = [tuple(group[key] for key in ("kind", "context", "target")) for group in groups]
            require(all(key[0] in ("context", "target") and all(isinstance(value, str) and value for value in key) for key in keys), "Invalid diagnostic group key")
            require(len(set(keys)) == len(keys) and {key[0] for key in keys} == {"context", "target"}, "Duplicate or incomplete diagnostic groups")
            require(set(quality["kind_metrics"]) == {"context", "target"}, "Both diagnostic metric kinds required")
            for kind, metrics in quality["kind_metrics"].items():
                for key, value in metrics.items():
                    require(type(value) in (int, float) and math.isfinite(value), f"Invalid {kind} {key}")
                    if key.endswith("centroid_mse"):
                        require(value >= 0, "Negative centroid MSE")
            comparable[arm] = {key: provenance[key] for key in
                               ("cache_sha256", "embeddings_sha256", "code_sha256", "upstream", "embedding_receipt")}
            comparable[arm].update(training_config={key: value for key, value in training.items() if key != "feature_mode"},
                                   model_config={key: value for key, value in model.items() if key != "feature_mode"},
                                   diagnostic_group_keys=sorted(keys),
                                   diagnostic_settings={key: rollout.get(key) for key in
                                                        ("ode_solver", "ode_steps", "cells_per_group", "seed", "noise_std")})
        except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError) as error:
            result["errors"].append(f"{arm}: {error}")
    if len(comparable) == 3:
        for arm in ARMS[1:]:
            for key in comparable["true"]:
                if comparable[arm][key] != comparable["true"][key]:
                    result["errors"].append(f"{arm}: {key} differs from true arm")
    if not result["errors"]:
        better = True
        for arm in ARMS[1:]:
            differences = {}
            for kind in ("context", "target"):
                differences[kind] = {metric: result["arms"]["true"]["public_metrics"][kind][f"model_{metric}"] -
                                             result["arms"][arm]["public_metrics"][kind][f"model_{metric}"]
                                     for metric in METRICS}
                better = better and all(value < 0 for value in differences[kind].values())
            result["true_minus_comparator"][arm] = differences
        result["status"] = "pass"
        if better:
            result["lingshu_benefit_claim"] = "true features improve all four metrics over both comparators; development comparison only, not independent test"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/lingshu_scdfm/effect_v2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    inputs = {args.root / arm / "fit" / name for arm in ARMS for name in ("training_summary.json", "training_provenance.json")}
    if args.output.resolve() in {path.resolve() for path in inputs}:
        parser.error("--output must not overwrite an input")
    result = compare(args.root)
    serialized = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent,
                                         prefix=f".{args.output.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(serialized)
        os.replace(temporary, args.output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    print(serialized, end="")
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
