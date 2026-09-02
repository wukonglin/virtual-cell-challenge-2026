#!/usr/bin/env python3
"""Run the gate-integrated Phase-1 V7 portable trainer smoke.

This entrypoint is intentionally narrower than compact model training. It
authenticates the repository-tracked gate, reconstructs the complete allowed
row sequence from one registered source, and authorizes every fitted stage
before reading expression values or constructing a model. The default mode
then performs two deterministic replicas of a small CFM/MMD optimizer step
using real allowed Jurkat rows and continuous Arc target features.

Passing this smoke proves trainer/gate integration only. It never creates a
challenge prediction or an official-submission artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import secrets
import socket
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from scdfm_portable_training_gate import (
    PortableTrainingGateError,
    canonical_sha256,
    is_sha256,
    load_portable_training_gate,
    open_regular_no_follow,
    read_regular_bytes,
    require,
    resolve_repository_path,
)


RECEIPT_SCHEMA = "vcc-scdfm-portable-trainer-smoke-receipt-v1"
DEFAULT_LOCK = Path(
    "artifacts/scdfm/v7/training/portable_training_gate_lock.json"
)
DEFAULT_MANIFEST = Path("artifacts/scdfm/v7/splits/jurkat_non_harm.json")
DEFAULT_PREFLIGHT = Path(
    "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
)
DEFAULT_CONFIG = Path("configs/scdfm/vcc2026_v7_gamma1.toml")
DEFAULT_SOURCE = Path(
    "dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--lock", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--preflight", type=Path, default=DEFAULT_PREFLIGHT)
    parser.add_argument("--split-configuration", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--expected-lock-sha256", required=True)
    parser.add_argument("--expected-trainer-sha256", required=True)
    parser.add_argument("--expected-gate-consumer-sha256", required=True)
    parser.add_argument("--launcher", type=Path)
    parser.add_argument("--expected-launcher-sha256")
    parser.add_argument("--expected-git-commit")
    parser.add_argument("--dataset", default="nadig_jurkat")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gene-count", type=int, default=64)
    parser.add_argument("--optimizer-steps", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args(argv)


def _resolved(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _descriptor(path: Path, label: str) -> dict[str, Any]:
    _, descriptor = read_regular_bytes(path, label)
    return descriptor


def _authenticate_code_file(
    path: Path,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    require(is_sha256(expected_sha256), f"A lowercase expected {label} SHA-256 is required")
    descriptor = _descriptor(path, label)
    require(
        descriptor["sha256"] == expected_sha256,
        f"{label} differs from the independently expected SHA-256",
    )
    return descriptor


def _require_expected_git_state(root: Path, expected_commit: str) -> dict[str, Any]:
    require(
        isinstance(expected_commit, str)
        and len(expected_commit) == 40
        and set(expected_commit).issubset("0123456789abcdef"),
        "A lowercase 40-character expected Git commit is required",
    )
    state = _git_state(root)
    require(state["available"] is True, "Git state is unavailable")
    require(state["commit"] == expected_commit, "Git commit differs from the submitted commit")
    require(state["dirty"] is False, "H100 evidence requires a clean Git worktree")
    return state


def _load_registered_config(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    raw, descriptor = read_regular_bytes(path, "split-audit configuration")
    try:
        payload = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise PortableTrainingGateError(f"Invalid split-audit TOML: {error}") from error
    return payload, descriptor


def _select_device(requested: str, require_cuda: bool, require_h100: bool):
    import torch

    if require_h100:
        require_cuda = True
    if requested == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda":
        require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        require(index < torch.cuda.device_count(), "Requested CUDA device is unavailable")
        device = torch.device("cuda", index)
    require(not require_cuda or device.type == "cuda", "This run requires CUDA")
    if require_h100:
        require(
            "H100" in torch.cuda.get_device_name(device).upper(),
            f"This run requires an H100, found {torch.cuda.get_device_name(device)}",
        )
    return device


def _load_selected_target_features(
    path: Path,
    *,
    expected_size: int,
    expected_sha256: str,
    expected_dimension: int,
    requested_targets: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restricted-load only requested feature vectors from authenticated bytes."""

    import torch

    handle, opened = open_regular_no_follow(path, "Arc target-feature artifact")
    with handle:
        digest = hashlib.sha256()
        while block := handle.read(8 << 20):
            digest.update(block)
        observed_sha256 = digest.hexdigest()
        require(opened.st_size == expected_size, "Target-feature artifact size changed")
        require(observed_sha256 == expected_sha256, "Target-feature artifact SHA-256 changed")
        handle.seek(0)
        try:
            payload = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise PortableTrainingGateError(
                f"Restricted target-feature load failed: {error}"
            ) from error
        require(type(payload) is dict, "Target-feature payload root is not a plain dict")
        selected: dict[str, Any] = {}
        for target in sorted(set(requested_targets)):
            value = payload.get(target)
            require(type(value) is torch.Tensor, f"Missing target feature: {target}")
            require(
                value.device.type == "cpu"
                and value.dtype == torch.float32
                and tuple(value.shape) == (expected_dimension,),
                f"Invalid target feature: {target}",
            )
            require(
                bool(torch.isfinite(value).all()) and bool(torch.count_nonzero(value)),
                f"Non-finite or zero target feature: {target}",
            )
            selected[target] = value.detach().clone()
        del payload
        handle.seek(0)
        second_digest = hashlib.sha256()
        while block := handle.read(8 << 20):
            second_digest.update(block)
        require(
            second_digest.hexdigest() == observed_sha256,
            "Target-feature bytes changed during restricted loading",
        )
        after = os.fstat(handle.fileno())
        require(
            (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
                opened.st_ctime_ns,
            )
            == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ),
            "Target-feature artifact identity changed during loading",
        )
    descriptor = {
        "filename": path.name,
        "size_bytes": opened.st_size,
        "sha256": observed_sha256,
        "restricted_weights_only_load": True,
        "selected_target_count": len(selected),
        "embedding_dimension": expected_dimension,
        "selected_target_list_sha256": canonical_sha256(sorted(selected)),
    }
    return selected, descriptor


def _library_normalize_log1p(values):
    import torch

    depth = values.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.log1p(values / depth * 10_000.0)


def _pairwise_squared_distance(left, right):
    return (left[:, None, :] - right[None, :, :]).square().sum(dim=-1)


def _multi_kernel_mmd2(generated, target, scales: Sequence[float]):
    import torch

    require(generated.shape == target.shape, "MMD endpoint shapes differ")
    require(generated.shape[0] >= 2, "Unbiased MMD requires at least two cells")
    d_generated = _pairwise_squared_distance(generated, generated)
    d_target = _pairwise_squared_distance(target, target)
    d_cross = _pairwise_squared_distance(generated, target)
    mask = ~torch.eye(target.shape[0], dtype=torch.bool, device=target.device)
    median = d_target[mask].detach().median().clamp_min(1e-12)
    estimates = []
    count = target.shape[0]
    for scale in scales:
        denominator = 2.0 * median * float(scale) + 1e-12
        left = torch.exp(-d_generated / denominator)
        right = torch.exp(-d_target / denominator)
        cross = torch.exp(-d_cross / denominator)
        estimates.append(
            (left.sum() - left.diagonal().sum()) / (count * (count - 1))
            + (right.sum() - right.diagonal().sum()) / (count * (count - 1))
            - 2.0 * cross.mean()
        )
    return torch.stack(estimates).mean()


def _state_dict_sha256(model) -> str:
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        name_bytes = name.encode("utf-8")
        metadata = json.dumps(
            {"dtype": str(value.dtype), "shape": list(value.shape)},
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        digest.update(len(name_bytes).to_bytes(8, "little"))
        digest.update(name_bytes)
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _train_replica(
    *,
    target,
    control,
    target_features,
    hidden_size: int,
    mmd_weight: float,
    mmd_scales: Sequence[float],
    seed: int,
    steps: int,
    device,
) -> dict[str, Any]:
    import torch
    from torch import nn

    class PortableVelocity(nn.Module):
        def __init__(self, genes: int, feature_dimension: int) -> None:
            super().__init__()
            self.x = nn.Linear(genes, hidden_size)
            self.control = nn.Linear(genes, hidden_size)
            self.target = nn.Linear(feature_dimension, hidden_size, bias=False)
            self.time = nn.Linear(1, hidden_size)
            self.output = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, genes),
            )

        def forward(self, x_t, context, condition, time):
            hidden = (
                self.x(x_t)
                + self.control(context)
                + self.target(condition)
                + self.time(time[:, None])
            )
            return self.output(hidden)

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    try:
        model = PortableVelocity(target.shape[1], target_features.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
        losses: list[dict[str, float]] = []
        for step in range(steps):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + step)
            noise = torch.randn(
                target.shape,
                generator=generator,
                dtype=torch.float32,
            ).to(device)
            time = torch.linspace(
                0.05,
                0.95,
                target.shape[0],
                dtype=torch.float32,
                device=device,
            )
            time_column = time[:, None]
            x_t = (1.0 - time_column) * noise + time_column * target
            reference_velocity = target - noise
            optimizer.zero_grad(set_to_none=True)
            predicted_velocity = model(
                x_t,
                control,
                target_features,
                time,
            )
            cfm = torch.nn.functional.mse_loss(
                predicted_velocity, reference_velocity
            )
            endpoint = x_t + (1.0 - time_column) * predicted_velocity
            mmd = _multi_kernel_mmd2(endpoint, target, mmd_scales)
            total = cfm + float(mmd_weight) * mmd
            total.backward()
            gradients = [
                parameter.grad
                for parameter in model.parameters()
                if parameter.requires_grad
            ]
            require(
                gradients and all(
                    gradient is not None and bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                ),
                "Portable trainer produced missing or non-finite gradients",
            )
            optimizer.step()
            require(
                all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters()),
                "Portable trainer produced non-finite parameters",
            )
            losses.append(
                {
                    "cfm_velocity_mse": float(cfm.detach().cpu()),
                    "multi_kernel_rbf_mmd2": float(mmd.detach().cpu()),
                    "total": float(total.detach().cpu()),
                }
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return {
            "losses": losses,
            "model_state_sha256": _state_dict_sha256(model),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
        }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def _git_state(root: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {
            "available": True,
            "commit": commit,
            "dirty": bool(status),
            "changed_path_count": len(status),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"available": False, "commit": None, "dirty": None}


def _write_json_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path, follow_symlinks=False)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite trainer receipt: {path}") from error
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run(args: argparse.Namespace) -> dict[str, Any]:
    modules_at_entry = frozenset(sys.modules)
    require(args.batch_size >= 2, "Batch size must be at least two")
    require(args.gene_count >= 2, "Gene count must be at least two")
    require(args.optimizer_steps >= 1, "Optimizer steps must be positive")
    root = Path(os.path.abspath(os.fspath(args.repository_root.expanduser())))
    trainer_path = Path(os.path.abspath(__file__))
    gate_consumer_path = trainer_path.with_name(
        "scdfm_portable_training_gate.py"
    )
    if args.require_h100:
        require(
            trainer_path
            == resolve_repository_path(
                root,
                "scripts/train_scdfm_v7_portable.py",
                "portable trainer",
            ),
            "Executed trainer path differs from the registered repository path",
        )
        require(
            gate_consumer_path
            == resolve_repository_path(
                root,
                "scripts/scdfm_portable_training_gate.py",
                "portable gate consumer",
            ),
            "Gate-consumer path differs from the registered repository path",
        )
    registered_code = {
        "trainer": _authenticate_code_file(
            trainer_path,
            args.expected_trainer_sha256,
            "portable trainer",
        ),
        "gate_consumer": _authenticate_code_file(
            gate_consumer_path,
            args.expected_gate_consumer_sha256,
            "portable gate consumer",
        ),
    }
    require(
        (args.launcher is None) == (args.expected_launcher_sha256 is None),
        "Launcher path and independently expected SHA-256 must be supplied together",
    )
    if args.launcher is not None:
        launcher_path = _resolved(root, args.launcher)
        registered_launcher_path = resolve_repository_path(
            root,
            "slurm/h100_train_scdfm_v7_portable_smoke.sbatch",
            "portable H100 launcher",
        )
        require(
            Path(os.path.abspath(launcher_path)) == registered_launcher_path,
            "Launcher path differs from the registered repository path",
        )
        registered_code["launcher"] = _authenticate_code_file(
            launcher_path,
            args.expected_launcher_sha256,
            "portable H100 launcher",
        )
    if args.require_h100:
        require(
            args.launcher is not None and args.expected_git_commit is not None,
            "H100 evidence requires an authenticated launcher and expected Git commit",
        )
    git_state_start = (
        _require_expected_git_state(root, args.expected_git_commit)
        if args.expected_git_commit is not None
        else _git_state(root)
    )
    lock_path = _resolved(root, args.lock)
    manifest_path = _resolved(root, args.manifest)
    preflight_path = _resolved(root, args.preflight)
    config_path = _resolved(root, args.split_configuration)
    source_path = _resolved(root, args.source)

    # Code and Git authentication above read no biological data. No model,
    # transform, graph, or optimizer exists before the complete portable gate
    # authenticates below.
    gate = load_portable_training_gate(
        repository_root=root,
        lock_path=lock_path,
        manifest_path=manifest_path,
        preflight_path=preflight_path,
        split_configuration_path=config_path,
        expected_lock_sha256=args.expected_lock_sha256,
    )
    config, config_descriptor = _load_registered_config(config_path)
    require(config_descriptor == dict(gate.split_configuration_descriptor), "Trainer config descriptor differs from gate")
    experiment_config = config.get("experiment", {})
    require(isinstance(experiment_config, dict), "Invalid [experiment] section")
    seed = int(experiment_config.get("seed", 20260901))

    with gate.open_h5_source(args.dataset, source_path) as source:
        # Complete row identity is checked in __enter__. Each fitted stage is
        # explicitly opened before its corresponding operation can occur.
        source.authorize_stage("normalization_fit")
        source.authorize_stage("feature_selection")
        source.authorize_stage("model_fit")
        allowed = source.allowed_source_row_indices
        authorization = {
            "dataset": args.dataset,
            "role": source.rule.role,
            "source": dict(source.rule.source),
            "allowed_row_count": len(allowed),
            "allowed_source_row_indices_sha256": canonical_sha256(list(allowed)),
            "excluded_row_count": len(source.excluded_source_row_indices),
            "excluded_source_row_indices_sha256": canonical_sha256(
                list(source.excluded_source_row_indices)
            ),
            "authorized_stages": list(source.authorized_stages),
            "expression_values_read": not args.preflight_only,
        }
        if args.preflight_only:
            treated_labels: tuple[str, ...] = ()
            training = None
            feature_descriptor = None
            selected_rows = None
            device_metadata = None
        else:
            import torch

            controls = [
                index
                for index in allowed
                if source.labels[index] == "non-targeting"
            ]
            treated = [
                index
                for index in allowed
                if source.labels[index] != "non-targeting"
            ]
            require(
                len(controls) >= args.batch_size,
                "Not enough authorized controls",
            )
            require(
                len(treated) >= args.batch_size,
                "Not enough authorized treated cells",
            )
            control_rows = tuple(controls[: args.batch_size])
            treated_rows = tuple(treated[: args.batch_size])
            treated_labels = tuple(source.labels[index] for index in treated_rows)
            control_labels = tuple(source.labels[index] for index in control_rows)
            model_config = config.get("model")
            feature_config = config.get(
                "target_conditioning_artifact_authentication"
            )
            require(isinstance(model_config, dict), "Missing [model] section")
            require(
                isinstance(feature_config, dict),
                "Missing target-feature artifact registration",
            )
            target_feature_path = resolve_repository_path(
                root,
                feature_config["artifact"],
                "target-feature artifact",
            )
            features, feature_descriptor = _load_selected_target_features(
                target_feature_path,
                expected_size=int(feature_config["artifact_size_bytes"]),
                expected_sha256=str(feature_config["artifact_sha256"]),
                expected_dimension=int(feature_config["embedding_dimension"]),
                requested_targets=treated_labels,
            )
            genes = tuple(range(args.gene_count))
            treated_values = source.read_dense_expression(
                treated_rows,
                treated_labels,
                genes,
                "normalization_fit",
            )
            control_values = source.read_dense_expression(
                control_rows,
                control_labels,
                genes,
                "normalization_fit",
            )
            device = _select_device(
                args.device,
                require_cuda=args.require_cuda,
                require_h100=args.require_h100,
            )
            treated_tensor = _library_normalize_log1p(
                torch.from_numpy(treated_values).to(device)
            )
            control_tensor = _library_normalize_log1p(
                torch.from_numpy(control_values).to(device)
            )
            target_tensor = torch.stack(
                [features[label] for label in treated_labels]
            ).to(device)
            replica_one = _train_replica(
                target=treated_tensor,
                control=control_tensor,
                target_features=target_tensor,
                hidden_size=int(model_config["hidden_size"]),
                mmd_weight=float(model_config["mmd_weight"]),
                mmd_scales=tuple(model_config["mmd_kernel_scales"]),
                seed=seed,
                steps=args.optimizer_steps,
                device=device,
            )
            replica_two = _train_replica(
                target=treated_tensor,
                control=control_tensor,
                target_features=target_tensor,
                hidden_size=int(model_config["hidden_size"]),
                mmd_weight=float(model_config["mmd_weight"]),
                mmd_scales=tuple(model_config["mmd_kernel_scales"]),
                seed=seed,
                steps=args.optimizer_steps,
                device=device,
            )
            restart_identical = replica_one == replica_two
            training = {
                "replica_one": replica_one,
                "replica_two": replica_two,
                "deterministic_restart_identical": restart_identical,
            }
            selected_rows = {
                "treated_source_row_indices": list(treated_rows),
                "treated_source_row_indices_sha256": canonical_sha256(
                    list(treated_rows)
                ),
                "treated_labels": list(treated_labels),
                "treated_label_sequence_sha256": canonical_sha256(
                    list(treated_labels)
                ),
                "control_source_row_indices": list(control_rows),
                "control_source_row_indices_sha256": canonical_sha256(
                    list(control_rows)
                ),
                "gene_indices": list(genes),
            }
            device_metadata = {
                "type": device.type,
                "cuda_available": torch.cuda.is_available(),
                "name": (
                    torch.cuda.get_device_name(device)
                    if device.type == "cuda"
                    else "cpu"
                ),
                "torch": torch.__version__,
                "torch_cuda_runtime": torch.version.cuda,
            }

    registered_code_end = {
        label: _descriptor(
            (
                trainer_path
                if label == "trainer"
                else gate_consumer_path
                if label == "gate_consumer"
                else launcher_path
            ),
            label.replace("_", " "),
        )
        for label in registered_code
    }
    require(
        registered_code_end == registered_code,
        "Registered code bytes changed during the trainer run",
    )
    git_state_end = (
        _require_expected_git_state(root, args.expected_git_commit)
        if args.expected_git_commit is not None
        else _git_state(root)
    )

    checks = {
        "independently_expected_code_sha256_verified": True,
        "registered_code_unchanged_during_run": True,
        "expected_clean_git_commit_verified_when_required": (
            args.expected_git_commit is None
            or (
                git_state_start["commit"] == args.expected_git_commit
                and git_state_end["commit"] == args.expected_git_commit
                and git_state_start["dirty"] is False
                and git_state_end["dirty"] is False
            )
        ),
        "expected_lock_sha256_verified": True,
        "tracked_manifest_and_preflight_cross_link_verified": True,
        "split_configuration_descriptor_verified": True,
        "complete_allowed_row_sequence_verified_before_fitted_stages": True,
        "trainer_introduced_no_split_builder_or_kmeans": (
            "scdfm_jurkat_non_harm" not in (set(sys.modules) - modules_at_entry)
            and not any(
                name == "sklearn" or name.startswith("sklearn.")
                for name in (set(sys.modules) - modules_at_entry)
            )
        ),
        "evaluation_only_role_cannot_fit": True,
        "selected_rows_exclude_sealed_targets": all(
            label not in gate.excluded_targets for label in treated_labels
        ),
        "continuous_target_features_authenticated_or_preflight_only": (
            args.preflight_only or feature_descriptor is not None
        ),
        "finite_optimizer_step_completed_or_preflight_only": (
            args.preflight_only
            or all(
                math.isfinite(value)
                for step in training["replica_one"]["losses"]
                for value in step.values()
            )
        ),
        "deterministic_restart_verified_or_preflight_only": (
            args.preflight_only
            or training["deterministic_restart_identical"] is True
        ),
        "no_official_submission_artifact_created": True,
    }
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "passed" if all(checks.values()) else "failed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "purpose": (
                "portable_gate_preflight"
                if args.preflight_only
                else "gate_integrated_real_data_optimizer_smoke"
            ),
            "phase": "v7_phase1_adapter",
            "compact_model_training_ready": False,
            "model_quality_evaluation": False,
            "state_anchor_consumed": False,
            "challenge_prediction_created": False,
            "official_submission_allowed": False,
        },
        "gate": {
            "lock": dict(gate.lock_descriptor),
            "manifest": dict(gate.manifest_descriptor),
            "preflight": dict(gate.preflight_descriptor),
            "split_configuration": dict(gate.split_configuration_descriptor),
        },
        "authorization": authorization,
        "selected_rows": selected_rows,
        "target_features": feature_descriptor,
        "training": training,
        "device": device_metadata,
        "checks": checks,
        "registered_code": registered_code,
        "git": {
            "start": git_state_start,
            "end": git_state_end,
            "independently_expected_commit": args.expected_git_commit,
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "python": platform.python_version(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "seed": seed,
        },
    }
    require(receipt["status"] == "passed", "Portable trainer smoke checks failed")
    return receipt


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    receipt = run(args)
    _write_json_no_overwrite(args.output_json, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
