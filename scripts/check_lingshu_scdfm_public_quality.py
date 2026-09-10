#!/usr/bin/env python3
"""Fail before generation if the already-selected model fails public diagnostics.

This credential-free check does not choose a model or evaluate challenge-treated
cells. It mirrors the empirical rollout criteria of the preupload audit; that
full provenance, generated-output and official-format audit is still required.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

SCHEMA = "vcc-lingshu-scdfm-public-quality-preflight-v1"
MAX_HIGH_CLAMP = 0.01


def check_public_quality(training_summary):
    result = {
        "schema": SCHEMA,
        "status": "fail",
        "errors": [],
        "training_summary_sha256": None,
        "metric_ratios": {},
        "kind_metrics": {},
        "safeguards": {"max_high_clamp": MAX_HIGH_CLAMP, "strict_improvement": True},
        "interpretation": "Public diagnostic gate only; not a challenge score guarantee or replacement for the full preupload audit",
        "ratio_interpretation": "model/control; null for zero denominator or nonfinite ratio. Signed unbiased MMD is compared directly, not by its ratio.",
    }
    errors = result["errors"]

    def require(ok, message):
        if not ok:
            errors.append(message)

    def finite(value, label, nonnegative=False):
        ok = type(value) in (int, float)
        try:
            ok = ok and math.isfinite(value)
        except OverflowError:
            ok = False
        if not ok:
            errors.append(f"{label} must be a finite number")
            return None
        if nonnegative and value < 0:
            errors.append(f"{label} must be nonnegative")
        return value

    try:
        payload = Path(training_summary).read_bytes()
        result["training_summary_sha256"] = hashlib.sha256(payload).hexdigest()

        def reject(value):
            raise ValueError(f"Nonfinite JSON value: {value}")

        summary = json.loads(payload, parse_constant=reject)
        if not isinstance(summary, dict):
            raise ValueError("Training summary must be a JSON object")
        require(summary.get("schema") == "vcc-lingshu-scdfm-training-summary-v1", "Wrong training-summary schema")
        require(summary.get("status") == "completed", "Training is incomplete")
        require(summary.get("challenge_treated_used") is False, "Challenge-treated input is forbidden")
        require(summary.get("public_model_quality_evaluated") is True, "Completed public diagnostics required")
        require(summary.get("official_challenge_metrics_evaluated") is False, "Official challenge metrics must remain unevaluated")
        finite(summary.get("best_score"), "best_score", nonnegative=True)
        best_step, last_step = summary.get("best_step"), summary.get("last_step")
        require(type(best_step) is int and type(last_step) is int and 0 < best_step <= last_step,
                "Invalid selected training step")

        rollout = summary.get("deployment_rollout_diagnostics")
        if not isinstance(rollout, dict):
            raise ValueError("Deployment rollout diagnostics must be a JSON object")
        require(rollout.get("role") == "diagnostic_only_after_checkpoint_selection", "Rollout must be diagnostic-only after checkpoint selection")
        require(rollout.get("checkpoint_selection_affected") is False, "Rollout must not affect checkpoint selection")
        require(rollout.get("challenge_treated_used") is False, "Rollout must use public data only")
        require(rollout.get("official_challenge_metrics_evaluated") is False, "Rollout must not evaluate official challenge metrics")
        require(type(rollout.get("ode_steps")) is int and rollout["ode_steps"] == 20, "Expected 20-step deployment rollout")
        kinds = rollout.get("kind_metrics")
        if not isinstance(kinds, dict):
            raise ValueError("Rollout kind_metrics must be a JSON object")
        require(set(kinds) == {"context", "target"}, "Exactly context and target rollout kinds required")
        for kind in ("context", "target"):
            metrics = kinds.get(kind)
            if not isinstance(metrics, dict):
                errors.append(f"{kind} metrics must be a JSON object")
                continue
            ratios = {}
            clean_metrics = {}
            for metric in ("centroid_mse", "mmd2"):
                model_key, control_key = f"model_{metric}", f"control_{metric}"
                model = finite(metrics.get(model_key), f"{kind} {model_key}", nonnegative=metric == "centroid_mse")
                control = finite(metrics.get(control_key), f"{kind} {control_key}", nonnegative=metric == "centroid_mse")
                clean_metrics.update({model_key: model, control_key: control})
                ratios[metric] = None
                if model is not None and control is not None:
                    require(model < control, f"{kind} {metric} does not improve controls")
                    if control != 0:
                        ratio = model / control
                        ratios[metric] = ratio if math.isfinite(ratio) else None
            for clamp, maximum in (("high", MAX_HIGH_CLAMP), ("low", 1.0)):
                key = f"clamped_{clamp}_fraction"
                value = finite(metrics.get(key), f"{kind} {key}")
                clean_metrics[key] = value
                if value is not None:
                    require(0 <= value <= maximum,
                            f"{kind} {key} outside [0, {maximum}]")
            result["kind_metrics"][kind] = clean_metrics
            result["metric_ratios"][kind] = ratios
    except (OSError, ValueError, TypeError, KeyError) as error:
        errors.append(str(error))
    if not errors:
        result["status"] = "pass"
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.training_summary.resolve() == args.output.resolve():
        parser.error("--output must not overwrite --training-summary")
    result = check_public_quality(args.training_summary)
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
