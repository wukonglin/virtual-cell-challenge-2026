#!/usr/bin/env python3
"""Run a synthetic scDFM training-contract smoke test.

This program deliberately avoids challenge and public expression matrices.  It
checks only the registered V7 configuration boundary and a tiny, deterministic
PyTorch computation: linear conditional flow matching, multi-kernel RBF MMD,
and one finite backward/optimizer step.  It is not a model-quality evaluation
and it does not create a Virtual Cell Challenge submission artifact.
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
import stat
import sys
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn


SCHEMA = "vcc-scdfm-contract-smoke-receipt-v1"
DEFAULT_CONFIG = Path("configs/scdfm/vcc2026_v7_gamma1.toml")
DEFAULT_LAUNCHER = Path("slurm/h100_scdfm_contract_smoke.sbatch")
STATE_WEIGHT_KEY = "anchor.state_effect_weight"
SCDFM_MMD_WEIGHT_KEY = "model.mmd_weight"
MMD_ESTIMATOR = "unbiased_multi_kernel_gaussian_rbf"


class ContractSmokeError(RuntimeError):
    """Raised when the registered smoke-test contract is violated."""


@dataclass(frozen=True)
class SmokeConfig:
    """Validated subset of the V7 configuration used by this smoke test."""

    experiment_schema: str
    experiment_name: str
    seed: int
    official_submission_allowed: bool
    state_effect_weight: float
    state_anchor_immutable: bool
    hidden_size: int
    flow_path: str
    scdfm_mmd_weight: float
    mmd_kernel_scales: tuple[float, ...]
    absolute_scdfm_output_allowed: bool
    center_residual_by_context_target: bool


class TinyConditionalVelocity(nn.Module):
    """Minimal conditional velocity field for contract testing only."""

    def __init__(self, gene_count: int, hidden_size: int) -> None:
        super().__init__()
        input_size = gene_count * 3 + 1
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, gene_count),
        )

    def forward(
        self,
        x_t: Tensor,
        control: Tensor,
        perturbation_condition: Tensor,
        time: Tensor,
    ) -> Tensor:
        inputs = torch.cat(
            (x_t, control, perturbation_condition, time[:, None]), dim=1
        )
        return self.network(inputs)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractSmokeError(message)


def _section(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    _require(isinstance(value, dict), f"Missing TOML section [{name}]")
    return value


def _finite_number(value: Any, key: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{key} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{key} must be finite")
    return number


def read_and_describe_regular_file(
    path: Path, label: str
) -> tuple[dict[str, Any], bytes]:
    """Read and hash one regular file through one no-follow descriptor."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    parts = absolute.parts
    _require(len(parts) >= 2 and parts[0] == os.sep, f"Invalid {label} path")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    file_flags = os.O_RDONLY | getattr(os, "O_BINARY", 0)
    file_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor: int | None = None
    try:
        directory_descriptor = os.open(os.sep, directory_flags)
        for component in parts[1:-1]:
            next_descriptor = os.open(
                component,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = next_descriptor
        descriptor = os.open(
            parts[-1],
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as error:
        raise ContractSmokeError(f"Unable to open {label}: {error}") from error
    finally:
        if directory_descriptor is not None:
            os.close(directory_descriptor)

    with os.fdopen(descriptor, "rb") as handle:
        before = os.fstat(handle.fileno())
        _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while block := handle.read(1 << 20):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(handle.fileno())
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    _require(identity_before == identity_after, f"{label} changed while hashing")
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} size changed while reading")
    return (
        {
            "filename": absolute.name,
            "size_bytes": before.st_size,
            "sha256": digest.hexdigest(),
            "regular_file": True,
            "symlink_components_rejected": True,
            "bytes_read_and_hashed_from_same_descriptor": True,
        },
        payload,
    )


def describe_regular_file(path: Path, label: str) -> dict[str, Any]:
    """Hash one regular file while rejecting symlinks in every path component."""

    description, _ = read_and_describe_regular_file(path, label)
    return description


def parse_smoke_config(payload_bytes: bytes) -> SmokeConfig:
    """Parse and validate already authenticated V7 configuration bytes."""

    try:
        payload = tomllib.loads(payload_bytes.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ContractSmokeError(f"Invalid V7 configuration bytes: {error}") from error

    experiment = _section(payload, "experiment")
    anchor = _section(payload, "anchor")
    model = _section(payload, "model")

    experiment_schema = experiment.get("schema")
    experiment_name = experiment.get("name")
    seed = experiment.get("seed")
    _require(
        isinstance(experiment_schema, str) and experiment_schema,
        "experiment.schema must be a non-empty string",
    )
    _require(
        isinstance(experiment_name, str) and experiment_name,
        "experiment.name must be a non-empty string",
    )
    _require(
        isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0,
        "experiment.seed must be a nonnegative integer",
    )
    official_submission_allowed = experiment.get("official_submission_allowed")
    _require(
        official_submission_allowed is False,
        "The synthetic contract smoke must not authorize an official submission",
    )

    state_effect_weight = _finite_number(
        anchor.get("state_effect_weight"), STATE_WEIGHT_KEY
    )
    _require(
        state_effect_weight == 1.0,
        f"{STATE_WEIGHT_KEY} must be exactly 1.0",
    )
    state_anchor_immutable = anchor.get("immutable_during_scdfm_search")
    _require(
        state_anchor_immutable is True,
        "anchor.immutable_during_scdfm_search must be true",
    )

    hidden_size = model.get("hidden_size")
    _require(
        isinstance(hidden_size, int)
        and not isinstance(hidden_size, bool)
        and hidden_size > 0,
        "model.hidden_size must be a positive integer",
    )
    flow_path = model.get("flow_path")
    _require(
        flow_path == "linear_conditional_flow_matching",
        "model.flow_path must be linear_conditional_flow_matching",
    )
    scdfm_mmd_weight = _finite_number(
        model.get("mmd_weight"), SCDFM_MMD_WEIGHT_KEY
    )
    _require(scdfm_mmd_weight >= 0.0, f"{SCDFM_MMD_WEIGHT_KEY} must be nonnegative")
    absolute_scdfm_output_allowed = model.get("absolute_scdfm_output_allowed")
    _require(
        absolute_scdfm_output_allowed is False,
        "model.absolute_scdfm_output_allowed must be false",
    )
    center_residual_by_context_target = model.get(
        "center_residual_by_context_target"
    )
    _require(
        center_residual_by_context_target is True,
        "model.center_residual_by_context_target must be true",
    )

    raw_scales = model.get("mmd_kernel_scales")
    _require(isinstance(raw_scales, list), "model.mmd_kernel_scales must be a list")
    scales = tuple(
        _finite_number(value, f"model.mmd_kernel_scales[{index}]")
        for index, value in enumerate(raw_scales)
    )
    _require(len(scales) >= 2, "At least two MMD kernel scales are required")
    _require(all(scale > 0.0 for scale in scales), "MMD kernel scales must be positive")

    return SmokeConfig(
        experiment_schema=experiment_schema,
        experiment_name=experiment_name,
        seed=seed,
        official_submission_allowed=False,
        state_effect_weight=state_effect_weight,
        state_anchor_immutable=True,
        hidden_size=hidden_size,
        flow_path=flow_path,
        scdfm_mmd_weight=scdfm_mmd_weight,
        mmd_kernel_scales=scales,
        absolute_scdfm_output_allowed=False,
        center_residual_by_context_target=True,
    )


def load_smoke_config(path: Path) -> SmokeConfig:
    """Safely read, authenticate, and validate one V7 configuration file."""

    _, payload = read_and_describe_regular_file(path, "V7 configuration")
    return parse_smoke_config(payload)


def _pairwise_squared_distance(left: Tensor, right: Tensor) -> Tensor:
    return (left[:, None, :] - right[None, :, :]).square().sum(dim=-1)


def multi_kernel_rbf_mmd2(
    generated: Tensor,
    target: Tensor,
    scales: Sequence[float],
    *,
    epsilon: float = 1e-12,
) -> tuple[Tensor, Tensor]:
    """Return unbiased multi-kernel Gaussian RBF MMD squared and its median.

    The bandwidth policy follows the scDFM paper: the reference value is the
    median off-diagonal squared distance in the target batch and each kernel
    uses ``sigma_squared = scale * median``.  The unbiased estimator can be
    slightly negative for finite batches; finiteness, not nonnegativity, is the
    relevant contract check.
    """

    _require(generated.ndim == 2, "Generated MMD input must be rank two")
    _require(target.ndim == 2, "Target MMD input must be rank two")
    _require(
        generated.shape[1] == target.shape[1],
        "MMD feature dimensions must match",
    )
    _require(
        generated.shape[0] >= 2 and target.shape[0] >= 2,
        "Unbiased MMD requires at least two samples per distribution",
    )
    _require(len(scales) >= 2, "Multi-kernel MMD requires at least two scales")
    _require(
        all(math.isfinite(float(scale)) and float(scale) > 0.0 for scale in scales),
        "MMD scales must be finite and positive",
    )

    d_generated = _pairwise_squared_distance(generated, generated)
    d_target = _pairwise_squared_distance(target, target)
    d_cross = _pairwise_squared_distance(generated, target)
    target_off_diagonal = d_target[
        ~torch.eye(target.shape[0], dtype=torch.bool, device=target.device)
    ]
    reference_median = target_off_diagonal.detach().median().clamp_min(epsilon)

    generated_count = generated.shape[0]
    target_count = target.shape[0]
    estimates = []
    for scale in scales:
        sigma_squared = reference_median * float(scale)
        denominator = 2.0 * sigma_squared + epsilon
        kernel_generated = torch.exp(-d_generated / denominator)
        kernel_target = torch.exp(-d_target / denominator)
        kernel_cross = torch.exp(-d_cross / denominator)
        within_generated = (
            kernel_generated.sum() - kernel_generated.diagonal().sum()
        ) / (generated_count * (generated_count - 1))
        within_target = (
            kernel_target.sum() - kernel_target.diagonal().sum()
        ) / (target_count * (target_count - 1))
        estimates.append(within_generated + within_target - 2.0 * kernel_cross.mean())

    return torch.stack(estimates).mean(), reference_median


def _resolve_device(
    requested: str,
    *,
    require_cuda: bool,
    require_h100: bool,
) -> torch.device:
    if require_h100:
        require_cuda = True
    if requested == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)

    if device.type == "cuda":
        _require(torch.cuda.is_available(), "CUDA was requested but is unavailable")
        index = device.index if device.index is not None else torch.cuda.current_device()
        _require(index < torch.cuda.device_count(), f"CUDA device index is unavailable: {index}")
        device = torch.device("cuda", index)
    _require(
        not require_cuda or device.type == "cuda",
        "This run requires a CUDA device",
    )
    if require_h100:
        name = torch.cuda.get_device_name(device)
        _require("H100" in name.upper(), f"This run requires an H100, found: {name}")
    return device


def _device_metadata(device: torch.device) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    metadata: dict[str, Any] = {
        "type": device.type,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            {
                "index": index,
                "name": properties.name,
                "h100_detected": "H100" in properties.name.upper(),
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
                "multi_processor_count": properties.multi_processor_count,
            }
        )
    else:
        metadata.update({"index": None, "name": "cpu", "h100_detected": False})
    return metadata


def _software_metadata() -> dict[str, Any]:
    cudnn_version = (
        torch.backends.cudnn.version()
        if torch.backends.cudnn.is_available()
        else None
    )
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cudnn": cudnn_version,
    }


def _synthetic_batch(
    *,
    batch_size: int,
    gene_count: int,
    seed: int,
    state_effect_weight: float,
    device: torch.device,
) -> dict[str, Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    control = 0.4 + 0.4 * torch.rand(
        batch_size, gene_count, generator=generator, dtype=torch.float32
    )
    perturbation = torch.zeros(batch_size, gene_count, dtype=torch.float32)
    row_ids = torch.arange(batch_size)
    target_ids = row_ids.remainder(gene_count)
    perturbation[row_ids, target_ids] = -1.0

    state_effect = torch.zeros_like(control)
    state_effect[row_ids, target_ids] = -0.20 * control[row_ids, target_ids]
    downstream = torch.roll(perturbation, shifts=1, dims=1)
    state_effect = state_effect + 0.025 * downstream
    state_effect.requires_grad_(False)

    biological_residual = 0.015 * torch.randn(
        batch_size, gene_count, generator=generator, dtype=torch.float32
    )
    target = (
        control + state_effect_weight * state_effect + biological_residual
    ).clamp_min(0.0)
    noise = torch.randn(
        batch_size, gene_count, generator=generator, dtype=torch.float32
    )
    time = torch.linspace(0.05, 0.95, batch_size, dtype=torch.float32)

    return {
        "control": control.to(device),
        "perturbation": perturbation.to(device),
        "state_effect": state_effect.to(device),
        "target": target.to(device),
        "noise": noise.to(device),
        "time": time.to(device),
    }


def run_contract_smoke(
    *,
    config_path: Path,
    requested_device: str = "auto",
    require_cuda: bool = False,
    require_h100: bool = False,
    batch_size: int = 8,
    synthetic_gene_count: int = 32,
    launcher_path: Path | None = None,
) -> dict[str, Any]:
    """Execute one deterministic synthetic training step and return a receipt."""

    _require(batch_size >= 2, "batch_size must be at least two")
    _require(synthetic_gene_count >= 2, "synthetic_gene_count must be at least two")
    config_path = Path(os.path.abspath(os.fspath(config_path.expanduser())))
    implementation_path = Path(__file__)
    configuration_description, configuration_bytes = read_and_describe_regular_file(
        config_path,
        "V7 configuration",
    )
    registered_files = {
        "configuration": configuration_description,
        "implementation": describe_regular_file(
            implementation_path,
            "contract-smoke implementation",
        ),
    }
    if launcher_path is not None:
        registered_files["launcher"] = describe_regular_file(
            launcher_path,
            "contract-smoke Slurm launcher",
        )
    config = parse_smoke_config(configuration_bytes)
    device = _resolve_device(
        requested_device,
        require_cuda=require_cuda,
        require_h100=require_h100,
    )

    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    try:
        batch = _synthetic_batch(
            batch_size=batch_size,
            gene_count=synthetic_gene_count,
            seed=config.seed,
            state_effect_weight=config.state_effect_weight,
            device=device,
        )
        model = TinyConditionalVelocity(
            synthetic_gene_count, config.hidden_size
        ).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        optimizer.zero_grad(set_to_none=True)

        time_column = batch["time"][:, None]
        x_t = (1.0 - time_column) * batch["noise"] + time_column * batch["target"]
        reference_velocity = batch["target"] - batch["noise"]
        predicted_velocity = model(
            x_t,
            batch["control"],
            batch["perturbation"],
            batch["time"],
        )
        cfm_loss = torch.nn.functional.mse_loss(
            predicted_velocity, reference_velocity
        )
        predicted_endpoint = x_t + (1.0 - time_column) * predicted_velocity
        mmd_loss, reference_median = multi_kernel_rbf_mmd2(
            predicted_endpoint,
            batch["target"],
            config.mmd_kernel_scales,
        )
        weighted_mmd_loss = config.scdfm_mmd_weight * mmd_loss
        total_loss = cfm_loss + weighted_mmd_loss
        total_loss.backward()

        trainable_parameters = [
            parameter for parameter in model.parameters() if parameter.requires_grad
        ]
        gradients = [parameter.grad for parameter in trainable_parameters]
        gradients_present = bool(gradients) and all(
            gradient is not None for gradient in gradients
        )
        gradients_finite = gradients_present and all(
            bool(torch.isfinite(gradient).all().item())
            for gradient in gradients
            if gradient is not None
        )
        gradient_squared_norm = sum(
            float(gradient.detach().float().square().sum().item())
            for gradient in gradients
            if gradient is not None
        )
        gradient_l2_norm = math.sqrt(gradient_squared_norm)

        loss_values = {
            "cfm_velocity_mse": float(cfm_loss.detach().cpu().item()),
            "multi_kernel_rbf_mmd2": float(mmd_loss.detach().cpu().item()),
            "weighted_scdfm_mmd": float(weighted_mmd_loss.detach().cpu().item()),
            "total": float(total_loss.detach().cpu().item()),
            "mmd_reference_squared_distance_median": float(
                reference_median.detach().cpu().item()
            ),
        }
        optimizer.step()
        parameters_finite_after_step = all(
            bool(torch.isfinite(parameter).all().item())
            for parameter in trainable_parameters
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)

        finite_losses = all(math.isfinite(value) for value in loss_values.values())
        total_decomposition_matches = math.isclose(
            loss_values["total"],
            loss_values["cfm_velocity_mse"]
            + loss_values["weighted_scdfm_mmd"],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        mmd_weight_binding_matches = math.isclose(
            loss_values["weighted_scdfm_mmd"],
            config.scdfm_mmd_weight * loss_values["multi_kernel_rbf_mmd2"],
            rel_tol=1e-6,
            abs_tol=1e-7,
        )
        registered_files_after = {
            "configuration": describe_regular_file(config_path, "V7 configuration"),
            "implementation": describe_regular_file(
                implementation_path,
                "contract-smoke implementation",
            ),
        }
        if launcher_path is not None:
            registered_files_after["launcher"] = describe_regular_file(
                launcher_path,
                "contract-smoke Slurm launcher",
            )
        registered_files_unchanged = registered_files_after == registered_files

        checks = {
            "configuration_was_parsed_from_authenticated_descriptor_bytes": True,
            "config_disallows_official_submission": (
                config.official_submission_allowed is False
            ),
            "config_forbids_absolute_scdfm_output": (
                config.absolute_scdfm_output_allowed is False
            ),
            "config_requires_context_target_residual_centering": (
                config.center_residual_by_context_target is True
            ),
            "state_effect_weight_is_exactly_one": config.state_effect_weight == 1.0,
            "state_anchor_is_registered_immutable": config.state_anchor_immutable,
            "state_effect_tensor_is_frozen": not batch["state_effect"].requires_grad,
            "scdfm_mmd_weight_has_separate_config_binding": (
                STATE_WEIGHT_KEY != SCDFM_MMD_WEIGHT_KEY
            ),
            "multi_kernel_mmd_has_at_least_two_scales": (
                len(config.mmd_kernel_scales) >= 2
            ),
            "cfm_inputs_and_velocity_are_finite": all(
                bool(torch.isfinite(tensor).all().item())
                for tensor in (
                    x_t,
                    reference_velocity,
                    predicted_velocity,
                    predicted_endpoint,
                )
            ),
            "losses_are_finite": finite_losses,
            "weighted_mmd_uses_scdfm_mmd_weight": mmd_weight_binding_matches,
            "total_is_cfm_plus_weighted_scdfm_mmd": total_decomposition_matches,
            "all_trainable_gradients_are_present": gradients_present,
            "all_trainable_gradients_are_finite": gradients_finite,
            "gradient_l2_norm_is_positive": (
                math.isfinite(gradient_l2_norm) and gradient_l2_norm > 0.0
            ),
            "parameters_are_finite_after_optimizer_step": parameters_finite_after_step,
            "required_device_contract_is_met": (
                (not require_cuda or device.type == "cuda")
                and (
                    not require_h100
                    or "H100" in torch.cuda.get_device_name(device).upper()
                )
            ),
            "registered_config_implementation_and_launcher_are_unchanged": (
                registered_files_unchanged
            ),
        }

        device_metadata = _device_metadata(device)
        if device.type == "cuda":
            device_metadata["peak_memory_allocated_bytes"] = (
                torch.cuda.max_memory_allocated(device)
            )
            device_metadata["peak_memory_reserved_bytes"] = (
                torch.cuda.max_memory_reserved(device)
            )

        return {
            "schema": SCHEMA,
            "status": "passed" if all(checks.values()) else "failed",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": {
                "purpose": "synthetic_training_contract_smoke",
                "synthetic_data_only": True,
                "challenge_expression_read": False,
                "public_expression_read": False,
                "official_submission_artifact": False,
                "model_performance_evaluation": False,
            },
            "configuration": {
                "filename": config_path.name,
                "size_bytes": registered_files["configuration"]["size_bytes"],
                "sha256": registered_files["configuration"]["sha256"],
                "experiment_schema": config.experiment_schema,
                "experiment_name": config.experiment_name,
                "seed": config.seed,
                "flow_path": config.flow_path,
            },
            "registered_files": registered_files,
            "weight_bindings": {
                "frozen_state_anchor_scale": {
                    "config_key": STATE_WEIGHT_KEY,
                    "value": config.state_effect_weight,
                    "trainable": False,
                },
                "scdfm_distribution_regularizer": {
                    "config_key": SCDFM_MMD_WEIGHT_KEY,
                    "value": config.scdfm_mmd_weight,
                    "applies_only_to": "multi_kernel_rbf_mmd2",
                },
            },
            "training_contract": {
                "batch_size": batch_size,
                "synthetic_gene_count": synthetic_gene_count,
                "hidden_size": config.hidden_size,
                "optimizer": "SGD",
                "learning_rate": 1e-3,
                "optimizer_steps": 1,
                "mmd_estimator": MMD_ESTIMATOR,
                "mmd_kernel_scales": list(config.mmd_kernel_scales),
                "gradient_l2_norm": gradient_l2_norm,
            },
            "losses": loss_values,
            "checks": checks,
            "device": device_metadata,
            "software": _software_metadata(),
            "job": {
                "hostname": socket.gethostname(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            },
        }
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)


def _open_or_create_directory_no_follow(path: Path) -> int:
    """Open or create a directory using dirfds and reject every symlink."""

    absolute = Path(os.path.abspath(os.fspath(path.expanduser())))
    parts = absolute.parts
    _require(len(parts) >= 1 and parts[0] == os.sep, "Invalid receipt directory")
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(os.sep, flags)
    try:
        for component in parts[1:]:
            try:
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            except FileNotFoundError:
                os.mkdir(component, mode=0o755, dir_fd=descriptor)
                next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
    except Exception:
        os.close(descriptor)
        raise
    return descriptor


def write_json_atomic_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish canonical JSON through a symlink-safe dirfd without replacing."""

    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    _require(path.name not in {"", ".", ".."}, "Invalid receipt filename")
    parent_descriptor = _open_or_create_directory_no_follow(path.parent)
    temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
    temporary_created = False
    try:
        try:
            os.stat(path.name, dir_fd=parent_descriptor, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}")
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        flags |= getattr(os, "O_NOFOLLOW", 0)
        file_descriptor = os.open(
            temporary_name,
            flags,
            0o600,
            dir_fd=parent_descriptor,
        )
        temporary_created = True
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary_name,
                path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
        os.fsync(parent_descriptor)
    finally:
        if temporary_created:
            try:
                os.unlink(temporary_name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                pass
        os.close(parent_descriptor)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a synthetic PyTorch scDFM contract smoke; this is not a "
            "submission or model-performance evaluation."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--launcher",
        type=Path,
        default=None,
        help="Optional Slurm launcher whose exact bytes are bound into the receipt.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--require-h100", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--synthetic-gene-count", type=int, default=32)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = run_contract_smoke(
        config_path=args.config,
        requested_device=args.device,
        require_cuda=args.require_cuda,
        require_h100=args.require_h100,
        batch_size=args.batch_size,
        synthetic_gene_count=args.synthetic_gene_count,
        launcher_path=args.launcher,
    )
    write_json_atomic_no_overwrite(args.output_json, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    if receipt["status"] != "passed":
        raise ContractSmokeError("Synthetic scDFM contract checks did not all pass")
    return receipt


if __name__ == "__main__":
    try:
        main()
    except (ContractSmokeError, FileExistsError, OSError, ValueError) as error:
        print(f"scDFM contract smoke failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
