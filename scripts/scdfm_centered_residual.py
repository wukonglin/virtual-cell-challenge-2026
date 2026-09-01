#!/usr/bin/env python3
"""Build a provenance-gated continuous-space scDFM residual factor.

For a residual proposal ``P``, this tool uses ``R=P-mean_cells(P)``. For an
absolute endpoint ``F``, it uses ``R=(F-A)-mean_cells(F-A)`` relative to the
STATE anchor ``A``. The output is ``A + weight*R``. This is an intermediate
continuous-expression factor, not a raw-count or official-submission artifact.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import os
import re
import stat
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Iterator, Mapping, Sequence

import numpy as np


SCHEMA = "vcc-scdfm-centered-residual-v3"
ANCHOR_PROVENANCE_SCHEMA = "vcc-state-anchor-provenance-v2"
FLOW_PROVENANCE_SCHEMA = "vcc-scdfm-flow-provenance-v2"
ANCHOR_DATA_ROLE = "authenticated_state_anchor"
FLOW_DATA_ROLE = "model_generated_continuous_factor"
FLOW_SEMANTICS = ("residual_proposal", "absolute_endpoint")
DEFAULT_CONFIG = Path("configs/scdfm/vcc2026_v7_gamma1.toml")
REGISTERED_CENTROID_ATOL = 1e-10
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
NONNEGATIVE_REPRESENTATIONS = frozenset({"library_normalized_log1p"})


class ResidualContractError(ValueError):
    """Raised when an input violates the registered residual contract."""


@dataclass(frozen=True)
class AxisRegistration:
    path: Path
    sha256: str
    count: int
    values: tuple[str, ...]


@dataclass(frozen=True)
class ContextRegistration:
    path: Path
    sha256: str
    contexts: tuple[str, ...]
    count: int


@dataclass(frozen=True)
class Registration:
    experiment_schema: str
    experiment_seed: int
    registered_weights: tuple[float, ...]
    state_effect_weight: float
    p4_selection_path: Path
    p4_selection_sha256: str
    target_axis: AxisRegistration
    gene_axis: AxisRegistration
    context_manifest: ContextRegistration
    target_count: int
    gene_count: int
    cells_per_group: int
    expression_space: str
    centroid_atol: float
    config_sha256: str
    canonical_group_layout_sha256: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ResidualContractError(message)


def _lexical_absolute(path: Path) -> Path:
    """Return an absolute lexical path without resolving symbolic links."""

    return Path(os.path.abspath(os.fspath(path.expanduser())))


@contextlib.contextmanager
def _open_regular_no_follow(path: Path, label: str) -> Iterator[tuple[BinaryIO, Path]]:
    """Open one unchanged regular file without following its final symlink."""

    lexical = _lexical_absolute(path)
    try:
        initial = lexical.lstat()
    except OSError as error:
        raise ResidualContractError(f"Unable to inspect {label}: {lexical}: {error}") from error
    _require(not stat.S_ISLNK(initial.st_mode), f"{label} must not be a symbolic link: {lexical}")
    _require(stat.S_ISREG(initial.st_mode), f"{label} must be a regular file: {lexical}")
    no_follow = getattr(os, "O_NOFOLLOW", None)
    _require(
        isinstance(no_follow, int) and not isinstance(no_follow, bool) and no_follow != 0,
        f"Secure no-follow file opening is unavailable for {label}: {lexical}",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | no_follow
    try:
        descriptor = os.open(lexical, flags)
    except OSError as error:
        raise ResidualContractError(f"Unable to open {label}: {lexical}: {error}") from error
    handle = os.fdopen(descriptor, "rb")
    try:
        opened = os.fstat(handle.fileno())
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (initial.st_dev, initial.st_ino),
            f"{label} changed while being opened: {lexical}",
        )
        yield handle, lexical
        final = os.fstat(handle.fileno())
        _require(
            (final.st_dev, final.st_ino, final.st_size, final.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            f"{label} changed while being authenticated: {lexical}",
        )
    finally:
        handle.close()


def _sha256_handle(handle: BinaryIO, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    handle.seek(0)
    while block := handle.read(chunk_size):
        digest.update(block)
    handle.seek(0)
    return digest.hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    with _open_regular_no_follow(path, "file") as (handle, _):
        return _sha256_handle(handle, chunk_size)


def _read_regular_bytes(path: Path, label: str) -> tuple[Path, bytes, str]:
    with _open_regular_no_follow(path, label) as (handle, lexical):
        payload = handle.read()
        digest = hashlib.sha256(payload).hexdigest()
    return lexical, payload, digest


def sha256_array(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _sha256(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a SHA-256 string")
    digest = value.strip().lower()
    _require(
        SHA256_PATTERN.fullmatch(digest) is not None,
        f"{label} must contain exactly 64 hexadecimal characters",
    )
    return digest


def _config_path(value: object, config: Path, label: str) -> Path:
    _require(isinstance(value, str) and value, f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_absolute():
        return _lexical_absolute(path)
    candidates = [Path.cwd() / path]
    if len(config.parents) >= 3:
        candidates.append(config.parents[2] / path)
    candidates.append(config.parent / path)
    for candidate in candidates:
        try:
            candidate.lstat()
        except OSError:
            continue
        return _lexical_absolute(candidate)
    return _lexical_absolute(candidates[0])


def _axis(path: Path, column: str, expected_count: int) -> AxisRegistration:
    lexical, payload, digest = _read_regular_bytes(path, f"configured {column} axis")
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as error:
        raise ResidualContractError(f"Invalid {column} axis encoding: {error}") from error
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        _require(reader.fieldnames is not None and column in reader.fieldnames, f"Missing {column!r} column")
        values = [(row.get(column) or "").strip() for row in reader]
    _require(all(values), f"Configured {column} axis contains empty values")
    _require(len(values) == expected_count, f"Expected {expected_count} {column} rows, found {len(values)}")
    _require(len(set(values)) == len(values), f"Configured {column} axis contains duplicates")
    return AxisRegistration(lexical, digest, len(values), tuple(values))


def _context_manifest(
    path: Path,
    *,
    target_count: int,
    gene_count: int,
    cells_per_group: int,
) -> ContextRegistration:
    lexical, encoded, digest = _read_regular_bytes(path, "configured control manifest")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResidualContractError(f"Invalid control manifest: {error}") from error
    _require(isinstance(payload, dict), "Control manifest must be an object")
    contexts = payload.get("contexts")
    _require(
        isinstance(contexts, list)
        and contexts
        and all(isinstance(value, str) and value for value in contexts),
        "Control manifest contexts must be a non-empty ordered string list",
    )
    _require(len(set(contexts)) == len(contexts), "Control manifest contexts must be unique")
    _require(payload.get("n_genes") == gene_count, "Control manifest n_genes mismatch")
    _require(payload.get("n_constructs") == target_count, "Control manifest n_constructs mismatch")
    _require(payload.get("cells_per_pert") == cells_per_group, "Control manifest cells_per_pert mismatch")
    per_context = payload.get("per_context")
    _require(isinstance(per_context, dict), "Control manifest per_context table is missing")
    _require(
        list(per_context) == contexts,
        "Control manifest per_context order must exactly match contexts",
    )
    for context in contexts:
        record = per_context[context]
        _require(isinstance(record, dict), f"Invalid per-context record: {context}")
        _require(
            record.get("n_perturbations") == target_count,
            f"Control manifest perturbation count mismatch for {context}",
        )
    return ContextRegistration(
        path=lexical,
        sha256=digest,
        contexts=tuple(contexts),
        count=len(contexts),
    )


def _authenticate_p4(path: Path, expected_sha256: str, selected_arm: object) -> None:
    _, encoded, observed = _read_regular_bytes(path, "P4 selection receipt")
    _require(observed == expected_sha256, f"P4 selection SHA-256 mismatch: {observed}")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResidualContractError(f"Invalid P4 selection receipt: {error}") from error
    _require(isinstance(payload, dict) and payload.get("status") == "passed", "P4 selection did not pass")
    decision = payload.get("decision")
    _require(isinstance(decision, dict), "P4 selection has no decision")
    _require(selected_arm == "g100" and decision.get("selected_arm") == "g100", "P4 did not select g100")
    weight = decision.get("state_effect_weight")
    _require(
        isinstance(weight, (int, float)) and not isinstance(weight, bool) and float(weight) == 1.0,
        "P4 STATE effect weight must be exactly 1.0",
    )


def read_registration(path: Path) -> Registration:
    """Load and authenticate the complete V7 continuous-factor contract."""

    path, encoded, config_sha256 = _read_regular_bytes(path, "experiment config")
    try:
        payload = tomllib.loads(encoded.decode("utf-8"))
    except (UnicodeError, tomllib.TOMLDecodeError) as error:
        raise ResidualContractError(f"Invalid experiment config: {error}") from error
    try:
        experiment, anchor = payload["experiment"], payload["anchor"]
        challenge, model, search = payload["challenge"], payload["model"], payload["search"]
    except (KeyError, TypeError) as error:
        raise ResidualContractError("Config is missing a required V7 section") from error
    _require(all(isinstance(x, dict) for x in (experiment, anchor, challenge, model, search)), "V7 sections must be TOML tables")

    schema = experiment.get("schema")
    seed = experiment.get("seed")
    _require(isinstance(schema, str) and schema, "experiment.schema must be non-empty")
    _require(isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0, "experiment.seed must be nonnegative")
    _require(experiment.get("official_submission_allowed") is False, "official_submission_allowed must be false")
    _require(anchor.get("immutable_during_scdfm_search") is True, "STATE anchor must be immutable")
    state_weight = anchor.get("state_effect_weight")
    _require(
        isinstance(state_weight, (int, float)) and not isinstance(state_weight, bool)
        and math.isfinite(float(state_weight)) and float(state_weight) == 1.0,
        "STATE effect weight must be exactly 1.0",
    )
    _require(model.get("absolute_scdfm_output_allowed") is False, "absolute_scdfm_output_allowed must be false")
    _require(model.get("center_residual_by_context_target") is True, "center_residual_by_context_target must be true")
    expression_space = model.get("expression_space")
    _require(isinstance(expression_space, str) and expression_space, "expression_space must be non-empty")
    centroid_atol = model.get("centroid_atol")
    _require(
        isinstance(centroid_atol, (int, float)) and not isinstance(centroid_atol, bool)
        and float(centroid_atol) == REGISTERED_CENTROID_ATOL,
        f"centroid_atol must be exactly {REGISTERED_CENTROID_ATOL}",
    )
    weights_raw = search.get("scdfm_residual_weights")
    _require(isinstance(weights_raw, list) and weights_raw, "Residual weights must be a non-empty list")
    try:
        weights = tuple(float(value) for value in weights_raw)
    except (TypeError, ValueError) as error:
        raise ResidualContractError("Residual weights must be numeric") from error
    _require(all(math.isfinite(value) for value in weights), "Residual weights must be finite")
    _require(len(set(weights)) == len(weights), "Residual weights must be unique")
    _require(all(0.0 <= value <= 1.0 for value in weights) and 0.0 in weights, "Residual weights must be registered in [0,1] and include zero")
    try:
        target_count = int(challenge["target_count"])
        gene_count = int(challenge["gene_count"])
        cells = int(challenge["cells_per_context_target"])
    except (KeyError, TypeError, ValueError) as error:
        raise ResidualContractError("Invalid challenge dimensions") from error
    _require(target_count > 0 and gene_count > 0 and cells > 0, "Challenge dimensions must be positive")
    _require(challenge.get("measured_challenge_perturbation_data_allowed") is False, "Measured challenge perturbation data is forbidden")

    p4_digest = _sha256(anchor.get("selection_receipt_sha256"), "selection receipt SHA-256")
    p4_path = _config_path(anchor.get("selection_receipt"), path, "selection_receipt")
    _authenticate_p4(p4_path, p4_digest, anchor.get("selected_arm"))
    targets = _axis(_config_path(challenge.get("targets"), path, "targets"), "target_gene", target_count)
    genes = _axis(_config_path(challenge.get("gene_axis"), path, "gene_axis"), "gene_name", gene_count)
    contexts = _context_manifest(
        _config_path(challenge.get("control_manifest"), path, "control_manifest"),
        target_count=target_count,
        gene_count=gene_count,
        cells_per_group=cells,
    )
    group_layout_payload = {
        "ordering": "context_major_target_minor",
        "groups": [
            {"context": context, "target_gene": target}
            for context in contexts.contexts
            for target in targets.values
        ],
    }
    group_layout_sha256 = hashlib.sha256(
        json.dumps(
            group_layout_payload, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return Registration(
        schema, seed, weights, 1.0, p4_path, p4_digest, targets, genes,
        contexts, target_count, gene_count, cells, expression_space,
        REGISTERED_CENTROID_ATOL, config_sha256, group_layout_sha256,
    )


def _validate_array(name: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    _require(
        np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.complexfloating)
        and not np.issubdtype(array.dtype, np.bool_),
        f"{name} must have a real numeric dtype",
    )
    _require(array.ndim == 3, f"{name} must have shape [groups, cells, genes], found {array.shape}")
    _require(all(size > 0 for size in array.shape), f"{name} dimensions must be nonzero")
    converted = np.asarray(array, dtype=np.float64)
    _require(np.isfinite(converted).all(), f"{name} contains non-finite values")
    return converted


def build_centered_candidate(
    anchor: np.ndarray,
    proposed_flow: np.ndarray,
    *,
    weight: float,
    registered_weights: Sequence[float],
    state_effect_weight: float,
    flow_semantics: str = "residual_proposal",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Compose a centroid-preserving continuous-space residual factor."""

    anchor64 = _validate_array("anchor", anchor)
    flow64 = _validate_array("proposed flow", proposed_flow)
    _require(anchor64.shape == flow64.shape, "Anchor and proposed flow shapes differ")
    _require(flow_semantics in FLOW_SEMANTICS, f"Unsupported flow semantics: {flow_semantics}")
    selected_weight = float(weight)
    frozen_state_weight = float(state_effect_weight)
    registered = tuple(float(value) for value in registered_weights)
    _require(math.isfinite(selected_weight), "Residual weight must be finite")
    _require(math.isfinite(frozen_state_weight) and frozen_state_weight == 1.0, "STATE effect weight must be exactly 1.0")
    _require(selected_weight in registered, f"Residual weight {selected_weight} is not registered")

    residual = flow64 if flow_semantics == "residual_proposal" else flow64 - anchor64
    centered = residual - residual.mean(axis=1, keepdims=True, dtype=np.float64)
    centered -= centered.mean(axis=1, keepdims=True, dtype=np.float64)
    anchor_centroid = anchor64.mean(axis=1, dtype=np.float64)
    candidate = anchor64 + selected_weight * centered
    for _ in range(2):
        candidate -= (candidate.mean(axis=1, dtype=np.float64) - anchor_centroid)[:, None, :]
    error = float(np.max(np.abs(candidate.mean(axis=1, dtype=np.float64) - anchor_centroid)))
    _require(np.isfinite(candidate).all(), "Candidate contains non-finite values")
    _require(error <= REGISTERED_CENTROID_ATOL, f"Continuous-space centroid error {error} exceeds {REGISTERED_CENTROID_ATOL}")
    return candidate, {
        "centroid_atol": REGISTERED_CENTROID_ATOL,
        "continuous_space_centroid_identity_passed": True,
        "flow_semantics": flow_semantics,
        "max_abs_centroid_error": error,
        "residual_weight": selected_weight,
        "shape": list(candidate.shape),
        "state_effect_weight": frozen_state_weight,
    }


def load_array(
    path: Path, key: str | None = None
) -> tuple[np.ndarray, str | None, Path, str]:
    """Hash and load an array through one unchanged no-follow descriptor."""

    lexical = _lexical_absolute(path)
    _require(lexical.suffix.lower() in {".npy", ".npz"}, f"Unsupported array format: {lexical.suffix}")
    try:
        with _open_regular_no_follow(lexical, "array input") as (handle, lexical):
            digest = _sha256_handle(handle)
            if lexical.suffix.lower() == ".npy":
                _require(key is None, "NPY input cannot use an array key")
                value = np.asarray(np.load(handle, allow_pickle=False))
                return value, None, lexical, digest
            with np.load(handle, allow_pickle=False) as archive:
                names = tuple(archive.files)
                if key is None:
                    _require(len(names) == 1, f"NPZ has {len(names)} arrays; provide a key")
                    key = names[0]
                _require(key in names, f"NPZ key {key!r} is absent")
                value = np.array(archive[key], copy=True)
            return value, key, lexical, digest
    except ResidualContractError:
        raise
    except (OSError, ValueError) as error:
        raise ResidualContractError(f"Unable to load array input {lexical}: {error}") from error


def _identity(path: Path, array: np.ndarray, key: str | None, file_sha256: str | None = None) -> dict[str, Any]:
    return {
        "array_sha256": sha256_array(array),
        "dtype": str(array.dtype),
        "file_sha256": file_sha256 or sha256_file(path),
        "key": key,
        "shape": list(array.shape),
    }


def _authenticate_input_receipt(
    *,
    receipt_path: Path,
    expected_receipt_sha256: str,
    expected_schema: str,
    expected_role: str,
    artifact_path: Path,
    artifact_file_sha256: str,
    array: np.ndarray,
    array_key: str | None,
    registration: Registration,
    flow_semantics: str | None = None,
) -> dict[str, Any]:
    expected = _sha256(expected_receipt_sha256, "Expected provenance receipt SHA-256")
    receipt_path, encoded, observed = _read_regular_bytes(
        receipt_path, "input provenance receipt"
    )
    _require(observed == expected, "Provenance receipt SHA-256 mismatch")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ResidualContractError(f"Invalid provenance receipt: {error}") from error
    _require(isinstance(payload, dict) and payload.get("schema") == expected_schema, "Unexpected provenance receipt schema")
    _require(payload.get("status") == "passed", "Input provenance did not pass")
    _require(payload.get("data_role") == expected_role, "Input provenance data role is forbidden")
    _require(payload.get("measured_treated_data_used") is False, "Measured treated data is forbidden")
    _require(payload.get("official_submission_artifact") is False, "Input cannot be a submission artifact")
    _require(payload.get("experiment_seed") == registration.experiment_seed, "Input seed mismatch")
    _require(payload.get("representation") == registration.expression_space, "Input representation mismatch")
    axes = payload.get("axes")
    _require(isinstance(axes, dict), "Input provenance axes are missing")
    _require(axes.get("target_axis_sha256") == registration.target_axis.sha256, "Target-axis SHA-256 mismatch")
    _require(axes.get("gene_axis_sha256") == registration.gene_axis.sha256, "Gene-axis SHA-256 mismatch")
    _require(
        axes.get("context_manifest_sha256") == registration.context_manifest.sha256,
        "Context-manifest SHA-256 mismatch",
    )
    _require(
        payload.get("artifact")
        == _identity(artifact_path, array, array_key, artifact_file_sha256),
        "Input artifact identity mismatch",
    )
    group_layout_sha256 = _sha256(
        axes.get("canonical_group_layout_sha256"),
        "canonical group-layout SHA-256",
    )
    _require(
        group_layout_sha256 == registration.canonical_group_layout_sha256,
        "Canonical group-layout SHA-256 mismatch",
    )
    cell_axis_sha256 = _sha256(
        axes.get("cell_axis_identity_sha256"), "cell-axis identity SHA-256"
    )
    latent_axis_sha256 = _sha256(
        axes.get("latent_axis_identity_sha256"), "latent-axis identity SHA-256"
    )
    _require(cell_axis_sha256 != "0" * 64, "Cell-axis identity SHA-256 is a placeholder")
    _require(
        latent_axis_sha256 != "0" * 64,
        "Latent-axis identity SHA-256 is a placeholder",
    )
    if expected_schema == ANCHOR_PROVENANCE_SCHEMA:
        _require(payload.get("p4_selection_sha256") == registration.p4_selection_sha256, "Anchor P4 SHA-256 mismatch")
    else:
        _require(payload.get("flow_semantics") == flow_semantics, "Flow semantics receipt mismatch")
    return {
        "path": str(receipt_path),
        "sha256": observed,
        "schema": expected_schema,
        "data_role": expected_role,
        "canonical_group_layout_sha256": group_layout_sha256,
        "cell_axis_identity_sha256": cell_axis_sha256,
        "latent_axis_identity_sha256": latent_axis_sha256,
    }


def _temporary(path: Path, binary: bool = True):
    path.parent.mkdir(parents=True, exist_ok=True)
    return tempfile.NamedTemporaryFile(
        mode="w+b" if binary else "w",
        encoding=None if binary else "utf-8",
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False,
    )


def _stage_array(path: Path, array: np.ndarray, key: str) -> Path:
    _require(path.suffix.lower() in {".npy", ".npz"}, f"Unsupported output format: {path.suffix}")
    with _temporary(path) as handle:
        temporary = Path(handle.name)
        if path.suffix.lower() == ".npy":
            np.save(handle, array, allow_pickle=False)
        else:
            np.savez_compressed(handle, **{key: array})
        handle.flush(); os.fsync(handle.fileno())
    return temporary


def _stage_copy(path: Path, source: Path) -> Path:
    with _temporary(path) as output:
        temporary = Path(output.name)
        with _open_regular_no_follow(source, "zero-arm anchor") as (input_handle, _):
            while block := input_handle.read(8 << 20):
                output.write(block)
        output.flush(); os.fsync(output.fileno())
    return temporary


def _stage_json(path: Path, payload: Mapping[str, Any]) -> Path:
    with _temporary(path, binary=False) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    return temporary


def _publish(temporary: Path, destination: Path) -> None:
    try:
        os.link(temporary, destination)
    except FileExistsError as error:
        raise FileExistsError(f"Refusing to overwrite artifact: {destination}") from error
    finally:
        temporary.unlink(missing_ok=True)


def _new_destination(path: Path, label: str) -> Path:
    """Return a lexical destination and reject every existing inode or symlink."""

    lexical = _lexical_absolute(path)
    try:
        current = lexical.lstat()
    except FileNotFoundError:
        return lexical
    except OSError as error:
        raise ResidualContractError(f"Unable to inspect {label}: {lexical}: {error}") from error
    kind = "symbolic link" if stat.S_ISLNK(current.st_mode) else "existing path"
    raise FileExistsError(f"Refusing to overwrite {kind}: {lexical}")


def execute(
    *,
    anchor_path: Path,
    flow_path: Path,
    anchor_receipt_path: Path,
    anchor_receipt_sha256: str,
    flow_receipt_path: Path,
    flow_receipt_sha256: str,
    flow_semantics: str,
    output_path: Path,
    receipt_path: Path,
    config_path: Path = DEFAULT_CONFIG,
    weight: float,
    anchor_key: str | None = None,
    flow_key: str | None = None,
    output_key: str = "candidate",
) -> dict[str, Any]:
    """Authenticate, compose, and receipt one continuous-space factor."""

    anchor_path, flow_path = _lexical_absolute(anchor_path), _lexical_absolute(flow_path)
    config_path = _lexical_absolute(config_path)
    output_path = _new_destination(output_path, "output")
    receipt_path = _new_destination(receipt_path, "receipt")
    _require(flow_semantics in FLOW_SEMANTICS, f"Unsupported flow semantics: {flow_semantics}")
    _require(bool(output_key) and output_path != receipt_path, "Invalid output path or key")
    registration = read_registration(config_path)
    anchor, anchor_selected_key, anchor_path, anchor_file_sha256 = load_array(
        anchor_path, anchor_key
    )
    flow, flow_selected_key, flow_path, flow_file_sha256 = load_array(
        flow_path, flow_key
    )
    anchor64, flow64 = _validate_array("anchor", anchor), _validate_array("proposed flow", flow)
    _require(anchor64.shape == flow64.shape, "Anchor and proposed flow shapes differ")
    groups, cells, genes = anchor64.shape
    _require(cells == registration.cells_per_group, "Cell count does not match registration")
    _require(genes == registration.gene_count, "Gene count does not match registration")
    expected_groups = registration.target_count * registration.context_manifest.count
    _require(groups == expected_groups, f"Expected {expected_groups} context-target groups, found {groups}")
    if registration.expression_space in NONNEGATIVE_REPRESENTATIONS:
        _require(np.all(anchor64 >= 0), "Anchor is negative in log1p expression space")
        if flow_semantics == "absolute_endpoint":
            _require(np.all(flow64 >= 0), "Absolute endpoint is negative in log1p expression space")

    anchor_provenance = _authenticate_input_receipt(
        receipt_path=anchor_receipt_path, expected_receipt_sha256=anchor_receipt_sha256,
        expected_schema=ANCHOR_PROVENANCE_SCHEMA, expected_role=ANCHOR_DATA_ROLE,
        artifact_path=anchor_path, artifact_file_sha256=anchor_file_sha256,
        array=anchor, array_key=anchor_selected_key, registration=registration,
    )
    flow_provenance = _authenticate_input_receipt(
        receipt_path=flow_receipt_path, expected_receipt_sha256=flow_receipt_sha256,
        expected_schema=FLOW_PROVENANCE_SCHEMA, expected_role=FLOW_DATA_ROLE,
        artifact_path=flow_path, artifact_file_sha256=flow_file_sha256,
        array=flow, array_key=flow_selected_key,
        registration=registration, flow_semantics=flow_semantics,
    )
    for identity_name in (
        "canonical_group_layout_sha256",
        "cell_axis_identity_sha256",
        "latent_axis_identity_sha256",
    ):
        _require(
            anchor_provenance[identity_name] == flow_provenance[identity_name],
            f"Anchor/flow {identity_name} mismatch",
        )
    candidate, diagnostics = build_centered_candidate(
        anchor, flow, weight=weight, registered_weights=registration.registered_weights,
        state_effect_weight=registration.state_effect_weight, flow_semantics=flow_semantics,
    )
    if registration.expression_space in NONNEGATIVE_REPRESENTATIONS:
        _require(np.all(candidate >= 0), "Candidate is negative in log1p expression space")

    zero_arm = diagnostics["residual_weight"] == 0.0
    if zero_arm:
        _require(output_path.suffix.lower() == anchor_path.suffix.lower(), "Zero arm must retain the anchor container format")
        output_array, output_array_key = anchor, anchor_selected_key
    else:
        output_array = candidate
        output_array_key = output_key if output_path.suffix.lower() == ".npz" else None

    output_temp: Path | None = None
    receipt_temp: Path | None = None
    try:
        output_temp = _stage_copy(output_path, anchor_path) if zero_arm else _stage_array(output_path, candidate, output_key)
        output_digest = sha256_file(output_temp)
        if zero_arm:
            _require(output_digest == anchor_file_sha256, "Zero arm is not byte-identical to anchor")
        receipt: dict[str, Any] = {
            "schema": SCHEMA,
            "status": "passed",
            "scope": {
                "continuous_expression_factor_only": True,
                "raw_count_invariance_claimed": False,
                "official_submission_artifact": False,
                "requires_authenticated_count_renderer": True,
            },
            "contract": {
                "candidate_formula": "anchor + weight * centered_residual",
                "flow_semantics": flow_semantics,
                "registered_residual_weights": list(registration.registered_weights),
            },
            "diagnostics": {**diagnostics, "zero_arm_byte_identical_to_anchor": zero_arm},
            "registration": {
                "config_sha256": registration.config_sha256,
                "experiment_seed": registration.experiment_seed,
                "expression_space": registration.expression_space,
                "p4_selection_sha256": registration.p4_selection_sha256,
                "target_axis_sha256": registration.target_axis.sha256,
                "gene_axis_sha256": registration.gene_axis.sha256,
                "context_manifest_sha256": registration.context_manifest.sha256,
                "canonical_group_layout_sha256": registration.canonical_group_layout_sha256,
                "cell_axis_identity_sha256": anchor_provenance["cell_axis_identity_sha256"],
                "latent_axis_identity_sha256": anchor_provenance["latent_axis_identity_sha256"],
                "contexts": list(registration.context_manifest.contexts),
            },
            "inputs": {
                "anchor": {
                    **_identity(anchor_path, anchor, anchor_selected_key, anchor_file_sha256),
                    "provenance": anchor_provenance,
                },
                "proposed_flow": {
                    **_identity(flow_path, flow, flow_selected_key, flow_file_sha256),
                    "provenance": flow_provenance,
                },
            },
            "output": {
                **_identity(output_path, output_array, output_array_key, output_digest),
                "path": str(output_path),
                "publication_mode": "byte_copy_of_anchor" if zero_arm else "new_continuous_factor",
            },
        }
        receipt_temp = _stage_json(receipt_path, receipt)
        _publish(output_temp, output_path); output_temp = None
        _require(sha256_file(output_path) == output_digest, "Published output hash mismatch")
        try:
            _publish(receipt_temp, receipt_path); receipt_temp = None
        except Exception:
            output_path.unlink(missing_ok=True)
            raise
    finally:
        if output_temp is not None:
            output_temp.unlink(missing_ok=True)
        if receipt_temp is not None:
            receipt_temp.unlink(missing_ok=True)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--anchor-receipt", type=Path, required=True)
    parser.add_argument("--anchor-receipt-sha256", required=True)
    parser.add_argument("--flow-receipt", type=Path, required=True)
    parser.add_argument("--flow-receipt-sha256", required=True)
    parser.add_argument("--flow-semantics", choices=FLOW_SEMANTICS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--anchor-key")
    parser.add_argument("--flow-key")
    parser.add_argument("--output-key", default="candidate")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    receipt = execute(
        anchor_path=args.anchor, flow_path=args.flow,
        anchor_receipt_path=args.anchor_receipt, anchor_receipt_sha256=args.anchor_receipt_sha256,
        flow_receipt_path=args.flow_receipt, flow_receipt_sha256=args.flow_receipt_sha256,
        flow_semantics=args.flow_semantics, output_path=args.output, receipt_path=args.receipt,
        config_path=args.config, weight=args.weight, anchor_key=args.anchor_key,
        flow_key=args.flow_key, output_key=args.output_key,
    )
    print(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
