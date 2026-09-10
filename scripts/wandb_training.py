"""Allowlisted W&B scalar tracking with a durable local event journal.

No code, raw console, arbitrary config, expression, or model artifacts upload.
The pinned SDK initializes in an empty directory, not the experiment cwd.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import uuid

SDK_VERSION = "0.19.11"
FIELDS = {
    "step": "train/global_step", "loss": "train/loss", "cfm": "train/cfm",
    "mmd": "train/mmd", "group_effect_cfm_mse": "train/group_effect_cfm_mse",
    "gradient_norm": "train/gradient_norm", "learning_rate": "train/learning_rate",
    "elapsed_seconds": "runtime/elapsed_seconds",
    "validation_velocity_mse": "validation/velocity_mse",
    "selection_group_effect_mse": "validation/group_effect_mse", "best_step": "validation/best_step",
    "completed_repeat": "cv/completed_repeat", "outer_fold": "cv/outer_fold",
    "last_step": "train/last_step", "best_score": "validation/best_score",
}
NUMERIC_CONFIG = {"steps", "batch_size", "learning_rate", "seed", "validate_every", "hidden_size",
                  "heads", "layers", "dropout", "mmd_weight", "noise_std", "world_size",
                  "validation_batch_size", "max_validation_groups", "weight_decay",
                  "integration_steps", "shuffle_seed"}
HASH_CONFIG = {"entrypoint_sha256", "cache_sha256", "embeddings_sha256", "parent_checkpoint_sha256",
               "split_sha256", "contract_sha256", "source_manifest_sha256",
               "target_features_sha256", "context_transform_sha256", "diagnostic_contract_sha256"}
FLOW_DIAGNOSTICS = ("variance_ratio", "covariance_effective_rank", "library_size_error",
                    "zero_fraction", "duplicate_fraction", "negative_fraction")
SHIFT_DIAGNOSTICS = ("predicted_shift_rms", "observed_shift_rms", "shift_dot_mean", "shift_cosine")
ANCHOR_DIAGNOSTICS = ("projection_fraction", "projection_mean_absolute_correction",
                      "projection_max_absolute_correction", "projection_centroid_distortion_rms",
                      "raw_negative_fraction", "raw_negative_magnitude_mean", "raw_negative_rms",
                      "raw_model_centroid_mse", "raw_model_mmd2", "ntc_sham_centroid_mse", "ntc_sham_mmd2")
DIAGNOSTICS = ("model_centroid_mse", "control_centroid_mse", "model_mmd2", "control_mmd2",
               "clamped_low_fraction", "clamped_high_fraction", *FLOW_DIAGNOSTICS,
               *SHIFT_DIAGNOSTICS, *ANCHOR_DIAGNOSTICS)
BIOLOGICAL_FAMILIES = ("esm2", "lingshu", "fusion", "genept", "go", "genept_go")


def allowed_cv_arm(name):
    return (name in ("true", "constant", "zero", *BIOLOGICAL_FAMILIES)
            or bool(re.fullmatch(r"(?:shuffled_|esm2_shuffled_|lingshu_shuffled_|fusion_shuffled_lingshu_|genept_shuffled_|go_shuffled_|genept_go_shuffled_)[0-9]{8}", str(name))))


def finite_scalar(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def valid_diagnostic(key, value):
    """Bound new aggregate flow diagnostics without changing signed MMD."""
    if not finite_scalar(value):
        return False
    if key in FLOW_DIAGNOSTICS:
        return value >= 0 and (key not in ("zero_fraction", "duplicate_fraction", "negative_fraction") or value <= 1)
    if key in ("predicted_shift_rms", "observed_shift_rms"):
        return value >= 0
    if key == "shift_cosine":
        return -1 <= value <= 1
    if key in ANCHOR_DIAGNOSTICS:
        if key in ("raw_model_mmd2", "ntc_sham_mmd2"):
            return True  # Preserve signed numerical MMD estimates.
        return value >= 0 and (key not in ("projection_fraction", "raw_negative_fraction") or value <= 1)
    return True


def safe_label(value, field):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"Invalid {field}; use letters, numbers, dots, dashes, or underscores")
    return value


def safe_config(values):
    result = {}
    for key, value in values.items():
        if key in NUMERIC_CONFIG and finite_scalar(value):
            result[key] = value
        elif key in HASH_CONFIG and isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value):
            result[key] = value
        elif key == "feature_mode" and value in ("true", "shuffled", "constant"):
            result[key] = value
    return result


def scalar_metrics(payload):
    if not isinstance(payload, dict):
        return {}
    result = {}
    for source, target in FIELDS.items():
        value = payload.get(source)
        if finite_scalar(value):
            if source in ("step", "best_step", "last_step", "completed_repeat", "outer_fold"):
                if type(value) is not int or value < 0:
                    continue
            result[target] = value
    mse = payload.get("mse", {})
    if isinstance(mse, dict):
        for mode, value in mse.items():
            if allowed_cv_arm(mode) and finite_scalar(value):
                result[f"cv/mse/{mode}"] = value
    decision = payload.get("decision", {})
    if isinstance(decision, dict) and type(decision.get("conditioning_screen_passed")) is bool:
        result["quality/conditioning_screen_passed"] = int(decision["conditioning_screen_passed"])
    decisions = payload.get("decisions", {})
    if isinstance(decisions, dict):
        for family in BIOLOGICAL_FAMILIES:
            item = decisions.get(family, {})
            if isinstance(item, dict) and type(item.get("conditioning_screen_passed")) is bool:
                result[f"quality/{family}_conditioning_screen_passed"] = int(item["conditioning_screen_passed"])
    # Strict schema and exact bool: generic comparison status=pass is NOT quality.
    if payload.get("schema") == "vcc-lingshu-scdfm-public-quality-preflight-v1":
        if payload.get("status") in ("pass", "fail"):
            result["quality/public_gate_passed"] = int(payload["status"] == "pass")
    rollout = payload.get("deployment_rollout_diagnostics", payload)
    kinds = rollout.get("kind_metrics", {}) if isinstance(rollout, dict) else {}
    if isinstance(kinds, dict):
        for kind in ("context", "target"):
            values = kinds.get(kind, {})
            if isinstance(values, dict):
                for key in DIAGNOSTICS:
                    if valid_diagnostic(key, values.get(key)):
                        result[f"validation/{kind}/{key}"] = values[key]
    return result


def line_metrics(line):
    if len(line) > 65536 or not line.lstrip().startswith("{"):
        return {}
    try:
        return scalar_metrics(json.loads(line))
    except (ValueError, TypeError):
        return {}


def config_from_command(command):
    """Only typed known flags; never serialize raw argv, paths or secrets."""
    values = {}
    for index, arg in enumerate(command):
        if not arg.startswith("--"):
            continue
        flag, separator, inline = arg[2:].partition("=")
        key = flag.replace("-", "_")
        value = inline if separator else (command[index + 1] if index + 1 < len(command) else "")
        if key in NUMERIC_CONFIG:
            with contextlib.suppress(ValueError):
                numeric = float(value)
                values[key] = int(numeric) if math.isfinite(numeric) and numeric.is_integer() else numeric
        elif key == "feature_mode" or key in HASH_CONFIG:
            values[key] = value
    return safe_config(values)


def global_rank(environ=None):
    environ = os.environ if environ is None else environ
    for key in ("RANK", "SLURM_PROCID", "OMPI_COMM_WORLD_RANK"):
        if key in environ:
            rank = int(environ[key])
            if rank < 0:
                raise ValueError("Global rank must be nonnegative")
            return rank
    return 0


def child_environment():
    # Child training must not create a second W&B run or inherit its API key.
    env = {key: value for key, value in os.environ.items() if not key.startswith("WANDB_")}
    env.update(WANDB_MODE="disabled", WANDB_DISABLED="true", PYTHONUNBUFFERED="1")
    return env


class TrainingTracker:
    def __init__(self, output_dir, *, stage, project="virtual-cell-challenge-2026", entity=None,
                 group="lingshu-public", name=None, mode="offline", config=None, sdk=None):
        if stage not in ("pretrain", "posttrain", "validation") or mode not in ("offline", "online"):
            raise ValueError("Explicit stage and online/offline mode required")
        project, group = safe_label(project, "project"), safe_label(group, "group")
        if entity is not None:
            entity = safe_label(entity, "entity")
        if name is not None:
            name = safe_label(name, "name")
        if mode == "online" and (not entity or not os.environ.get("WANDB_API_KEY")):
            raise ValueError("Online mode requires explicit entity and WANDB_API_KEY in the environment; otherwise use offline")
        if sdk is None:
            if importlib.metadata.version("wandb") != SDK_VERSION:
                raise RuntimeError("Install the pinned requirements/tracking.txt in a separate environment")
            # Import only AFTER suppressing implicit external W&B configuration.
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.run_id = uuid.uuid4().hex[:12]
        self.run = None
        self.events = 0
        self.failures = 0
        self.record = {"schema": "vcc-training-tracking-v1", "run_id": self.run_id,
                       "stage": stage, "mode": mode, "project": project, "entity": entity,
                       "group": group, "sdk_version": SDK_VERSION,
                       "cloud_synced": False, "execution_status": "initializing",
                       "config": safe_config(config or {}), "raw_artifacts_uploaded": False,
                       "console_code_environment_capture": False}
        self.journal = (self.output_dir / "metrics.jsonl").open("x", buffering=1)
        clean_dir = self.output_dir / "sdk"
        clean_dir.mkdir(mode=0o700)
        original_cwd = Path.cwd()
        original_env = {key: value for key, value in os.environ.items() if key.startswith("WANDB_")}
        try:
            for key in original_env:
                os.environ.pop(key, None)
            if mode == "online":
                os.environ["WANDB_API_KEY"] = original_env["WANDB_API_KEY"]
            os.environ["WANDB_CONFIG_DIR"] = str(clean_dir / "settings")
            os.chdir(clean_dir)
            if sdk is None:
                import wandb as sdk
            if getattr(sdk, "run", None) is not None:
                raise RuntimeError("Use the standalone wrapper in a fresh process, not an existing W&B run")
            settings = sdk.Settings(console="off", disable_git=True, disable_code=True,
                save_code=False, disable_job_creation=True, x_disable_meta=True,
                x_disable_stats=True, x_disable_machine_info=True, x_save_requirements=False,
                sagemaker_disable=True, x_log_level=30, config_paths=(), launch=False,
                init_timeout=30, login_timeout=10, quiet=True, symlink=False)
            self.run = sdk.init(project=project, entity=entity, group=group, job_type=stage,
                name=name or f"{stage}-{self.run_id}", id=self.run_id, mode=mode,
                dir=str(clean_dir), config={"stage": stage, **self.record["config"]},
                settings=settings, save_code=False, resume="never")
            self.run.define_metric("train/global_step")
            self.run.define_metric("train/*", step_metric="train/global_step")
            self.run.define_metric("validation/*", step_metric="train/global_step")
            self.record["execution_status"] = "running"
            self._receipt()
        except BaseException:
            self.record["execution_status"] = "tracking_initialization_failed"
            self._receipt()
            self.journal.close()
            raise
        finally:
            os.chdir(original_cwd)
            for key in tuple(os.environ):
                if key.startswith("WANDB_"):
                    os.environ.pop(key, None)
            os.environ.update(original_env)

    def _receipt(self):
        temporary = self.output_dir / "tracking.json.tmp"
        with temporary.open("w") as handle:
            json.dump(self.record, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, self.output_dir / "tracking.json")

    def log(self, metrics):
        # Public callers must supply only metrics produced by the sanitizer.
        allowed = set(FIELDS.values()) | {"quality/conditioning_screen_passed", "quality/public_gate_passed"}
        allowed |= {f"quality/{family}_conditioning_screen_passed" for family in BIOLOGICAL_FAMILIES}
        allowed |= {f"validation/{kind}/{key}" for kind in ("context", "target") for key in DIAGNOSTICS}
        metrics = {key: value for key, value in metrics.items()
                   if (key in allowed or (key.startswith("cv/mse/") and allowed_cv_arm(key[7:])))
                   and finite_scalar(value)}
        steps = {FIELDS[key] for key in ("step", "best_step", "last_step", "completed_repeat", "outer_fold")}
        metrics = {key: value for key, value in metrics.items()
                   if key not in steps or (type(value) is int and value >= 0)}
        metrics = {key: value for key, value in metrics.items()
                   if not key.startswith("validation/context/") and not key.startswith("validation/target/")
                   or valid_diagnostic(key.rsplit("/", 1)[-1], value)}
        if not metrics:
            return
        self.events += 1
        self.journal.write(json.dumps({"event_index": self.events, **metrics}, allow_nan=False) + "\n")
        # Event index is monotonic; two records at the same optimizer step survive.
        try:
            self.run.log(metrics, step=self.events)
        except Exception:
            self.failures += 1

    def summary(self, payload):
        """Aggregate scalars and typed provenance only, never the raw report."""
        self.log(scalar_metrics(payload))
        if not isinstance(payload, dict):
            return
        config = safe_config(payload)
        for field in ("training_config", "model_config"):
            if isinstance(payload.get(field), dict):
                config.update(safe_config(payload[field]))
        self.record["config"].update(config)
        try:
            if config:
                self.run.config.update(config, allow_val_change=True)
        except Exception:
            self.failures += 1

    def finish(self, exit_code, *, interrupted=False):
        self.record.update(execution_status="interrupted" if interrupted else ("completed" if exit_code == 0 else "failed"),
                           training_exit_code=int(exit_code), event_count=self.events,
                           tracking_errors=self.failures, scientific_quality_inferred_from_exit=False)
        try:
            self.run.summary.update({"execution/exit_code": int(exit_code), "execution/interrupted": interrupted,
                                     "tracking/event_count": self.events, "tracking/log_errors": self.failures})
            self.run.finish(exit_code=int(exit_code))
            self.record["cloud_synced"] = self.record["mode"] == "online" and not self.failures
        except Exception:
            self.record["tracking_errors"] += 1
        finally:
            self.journal.flush()
            os.fsync(self.journal.fileno())
            self.journal.close()
            self._receipt()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()
