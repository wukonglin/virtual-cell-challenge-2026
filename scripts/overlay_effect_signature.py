#!/usr/bin/env python3
"""Overlay a partial effect signature onto a full-axis Bayesian fallback.

The Bayesian artifact defines the complete challenge target and gene axes. The
signature replaces values only at coordinates that are explicitly available
through its named axes and optional presence masks. Every other effect remains
byte-identical to the aligned Bayesian value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


SCHEMA = "vcc-effect-signature-overlay-v1"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def describe_file(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def describe_companion_metadata(npz_path: Path) -> dict[str, Any] | None:
    """Describe and cross-check a same-stem JSON producer report when present."""

    metadata_path = npz_path.with_suffix(".json")
    if not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid companion JSON: {metadata_path}") from error
    require(isinstance(metadata, dict), f"Companion JSON must contain an object: {metadata_path}")
    recorded_hashes: list[str] = []
    direct_hash = metadata.get("output_npz_sha256")
    if isinstance(direct_hash, str):
        recorded_hashes.append(direct_hash)
    provenance = metadata.get("provenance", {})
    provenance_output = (
        provenance.get("output_npz", {}) if isinstance(provenance, dict) else {}
    )
    if isinstance(provenance_output, dict) and isinstance(provenance_output.get("sha256"), str):
        recorded_hashes.append(provenance_output["sha256"])
    outputs = metadata.get("outputs", {})
    effect_output = outputs.get("effect_prior", {}) if isinstance(outputs, dict) else {}
    if isinstance(effect_output, dict) and isinstance(effect_output.get("sha256"), str):
        recorded_hashes.append(effect_output["sha256"])
    actual_hash = sha256_file(npz_path)
    require(
        all(value == actual_hash for value in recorded_hashes),
        f"Companion JSON records a different NPZ hash: {metadata_path}",
    )
    description = describe_file(metadata_path)
    description["schema"] = metadata.get("schema", metadata.get("artifact_type"))
    description["recorded_npz_hashes"] = recorded_hashes
    return description


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bayesian_fallback", type=Path)
    parser.add_argument("signature_prior", type=Path)
    parser.add_argument("challenge_targets", type=Path)
    parser.add_argument("challenge_genes", type=Path)
    parser.add_argument("output_npz", type=Path)
    parser.add_argument("output_json", type=Path)
    parser.add_argument("--target-column", default="target_gene")
    parser.add_argument("--gene-column", default="gene_name")
    return parser.parse_args()


def _string_axis(
    payload: Any,
    keys: Sequence[str],
    path: Path,
    *,
    required: bool = True,
) -> tuple[str, ...] | None:
    observed: list[tuple[str, tuple[str, ...]]] = []
    for key in keys:
        if key in payload:
            values = tuple(
                value.decode("utf-8")
                if isinstance(value, (bytes, np.bytes_))
                else str(value)
                for value in np.asarray(payload[key]).reshape(-1)
            )
            require(values and all(values), f"Empty value in {key} from {path}")
            require(len(values) == len(set(values)), f"Duplicate value in {key} from {path}")
            observed.append((key, values))
    if observed:
        reference_key, reference = observed[0]
        for key, values in observed[1:]:
            require(
                values == reference,
                f"Axis aliases {reference_key} and {key} disagree in {path}",
            )
        return reference
    require(not required, f"Missing one of {list(keys)} in {path}")
    return None


def read_named_axis(path: Path, preferred: str, aliases: Sequence[str]) -> list[str]:
    require(path.is_file(), f"Missing axis CSV: {path}")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    accepted = (preferred, *aliases)
    for column in accepted:
        if column in table.columns:
            values = table[column].astype(str).tolist()
            break
    else:
        raw = pd.read_csv(path, header=None, dtype=str, keep_default_na=False)
        require(raw.shape[1] == 1, f"Expected one column in {path}")
        values = raw.iloc[:, 0].astype(str).tolist()
        if values and values[0].strip().lower() in {name.lower() for name in accepted}:
            values = values[1:]
    require(values and all(values), f"Empty axis in {path}")
    require(len(values) == len(set(values)), f"Duplicate axis value in {path}")
    return values


@dataclass(frozen=True)
class EffectInput:
    path: Path
    effects: np.ndarray
    contexts: tuple[str, ...] | None
    targets: tuple[str, ...]
    genes: tuple[str, ...]
    target_mask: np.ndarray
    gene_mask: np.ndarray
    context_mask: np.ndarray | None
    coordinate_mask: np.ndarray | None


def _optional_boolean_mask(
    payload: Any,
    keys: Sequence[str],
    shape: tuple[int, ...],
    path: Path,
) -> np.ndarray | None:
    for key in keys:
        if key in payload:
            values = np.asarray(payload[key], dtype=bool)
            require(values.shape == shape, f"Bad {key} shape in {path}: {values.shape}")
            return values
    return None


def load_effect_input(path: Path) -> EffectInput:
    require(path.is_file(), f"Missing effect input: {path}")
    with np.load(path, allow_pickle=False) as payload:
        require("effects" in payload, f"Missing effects in {path}")
        effects = np.asarray(payload["effects"], dtype=np.float32)
        targets = _string_axis(payload, ("targets", "target_names"), path)
        genes = _string_axis(payload, ("genes", "gene_names"), path)
        assert targets is not None and genes is not None
        contexts = _string_axis(payload, ("contexts", "context_names"), path, required=False)
        require(effects.ndim in (2, 3), f"Effects must have two or three dimensions in {path}")
        if effects.ndim == 2:
            require(
                effects.shape == (len(targets), len(genes)),
                f"Effect shape mismatch in {path}: {effects.shape}",
            )
            require(contexts is None, f"Two-dimensional effects cannot define contexts in {path}")
        else:
            require(contexts is not None, f"Three-dimensional effects require contexts in {path}")
            require(
                effects.shape == (len(contexts), len(targets), len(genes)),
                f"Effect shape mismatch in {path}: {effects.shape}",
            )
        require(np.isfinite(effects).all(), f"Non-finite effect in {path}")
        target_mask = _optional_boolean_mask(
            payload,
            ("effect_target_mask", "target_mask", "available_target_mask"),
            (len(targets),),
            path,
        )
        gene_mask = _optional_boolean_mask(
            payload,
            ("effect_gene_mask", "gene_mask", "available_gene_mask"),
            (len(genes),),
            path,
        )
        context_mask = (
            None
            if contexts is None
            else _optional_boolean_mask(
                payload,
                ("effect_context_mask", "context_mask", "available_context_mask"),
                (len(contexts),),
                path,
            )
        )
        coordinate_mask = None
        for key in ("effect_mask", "available_effect_mask"):
            if key in payload:
                coordinate_mask = np.asarray(payload[key], dtype=bool)
                valid_shapes = {effects.shape, (len(targets), len(genes))}
                require(
                    coordinate_mask.shape in valid_shapes,
                    f"Bad {key} shape in {path}: {coordinate_mask.shape}",
                )
                break
    return EffectInput(
        path=path,
        effects=effects,
        contexts=contexts,
        targets=targets,
        genes=genes,
        target_mask=(
            np.ones(len(targets), dtype=bool) if target_mask is None else target_mask
        ),
        gene_mask=np.ones(len(genes), dtype=bool) if gene_mask is None else gene_mask,
        context_mask=context_mask,
        coordinate_mask=coordinate_mask,
    )


def align_fallback(
    fallback: EffectInput,
    challenge_targets: Sequence[str],
    challenge_genes: Sequence[str],
) -> np.ndarray:
    target_lookup = {name: index for index, name in enumerate(fallback.targets)}
    gene_lookup = {name: index for index, name in enumerate(fallback.genes)}
    missing_targets = [name for name in challenge_targets if name not in target_lookup]
    missing_genes = [name for name in challenge_genes if name not in gene_lookup]
    require(not missing_targets, f"Fallback misses challenge targets: {missing_targets[:8]}")
    require(not missing_genes, f"Fallback misses challenge genes: {missing_genes[:8]}")
    target_indices = np.asarray([target_lookup[name] for name in challenge_targets], dtype=np.int64)
    gene_indices = np.asarray([gene_lookup[name] for name in challenge_genes], dtype=np.int64)
    require(
        bool(fallback.target_mask[target_indices].all()),
        "Fallback target mask does not cover every challenge target",
    )
    require(
        bool(fallback.gene_mask[gene_indices].all()),
        "Fallback gene mask does not cover every challenge gene",
    )
    if fallback.effects.ndim == 2:
        if fallback.coordinate_mask is not None:
            require(
                bool(
                    fallback.coordinate_mask[
                        np.ix_(target_indices, gene_indices)
                    ].all()
                ),
                "Fallback effect mask does not cover every challenge coordinate",
            )
        return fallback.effects[np.ix_(target_indices, gene_indices)].copy()
    context_indices = np.arange(fallback.effects.shape[0], dtype=np.int64)
    if fallback.context_mask is not None:
        require(
            bool(fallback.context_mask.all()),
            "Fallback context mask does not cover every output context",
        )
    if fallback.coordinate_mask is not None:
        if fallback.coordinate_mask.ndim == 2:
            selected_mask = fallback.coordinate_mask[
                np.ix_(target_indices, gene_indices)
            ]
        else:
            selected_mask = fallback.coordinate_mask[
                np.ix_(context_indices, target_indices, gene_indices)
            ]
        require(
            bool(selected_mask.all()),
            "Fallback effect mask does not cover every challenge coordinate",
        )
    return fallback.effects[np.ix_(context_indices, target_indices, gene_indices)].copy()


def overlay_signature(
    fallback_effects: np.ndarray,
    fallback_contexts: tuple[str, ...] | None,
    challenge_targets: Sequence[str],
    challenge_genes: Sequence[str],
    signature: EffectInput,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Return overlaid effects, an exact write mask, and alignment statistics."""

    output = fallback_effects.astype(np.float32, copy=True)
    write_mask = np.zeros(output.shape, dtype=bool)
    target_lookup = {name: index for index, name in enumerate(challenge_targets)}
    gene_lookup = {name: index for index, name in enumerate(challenge_genes)}
    outside_targets = sorted(set(signature.targets) - set(challenge_targets))
    outside_genes = sorted(set(signature.genes) - set(challenge_genes))
    require(
        not outside_targets,
        f"Signature targets are outside challenge axis: {outside_targets[:8]}",
    )
    require(not outside_genes, f"Signature genes are outside challenge axis: {outside_genes[:8]}")
    signature_target_indices = np.asarray(
        [
            index
            for index, name in enumerate(signature.targets)
            if name in target_lookup and signature.target_mask[index]
        ],
        dtype=np.int64,
    )
    signature_gene_indices = np.asarray(
        [
            index
            for index, name in enumerate(signature.genes)
            if name in gene_lookup and signature.gene_mask[index]
        ],
        dtype=np.int64,
    )
    require(len(signature_target_indices) > 0, "Signature has no available challenge target")
    require(len(signature_gene_indices) > 0, "Signature has no available challenge gene")
    output_target_indices = np.asarray(
        [target_lookup[signature.targets[index]] for index in signature_target_indices],
        dtype=np.int64,
    )
    output_gene_indices = np.asarray(
        [gene_lookup[signature.genes[index]] for index in signature_gene_indices],
        dtype=np.int64,
    )

    changed_coordinates = 0
    written_by_context: dict[str, int] = {}
    if output.ndim == 2:
        require(signature.effects.ndim == 2, "A contextual signature needs a contextual fallback")
        source = signature.effects[np.ix_(signature_target_indices, signature_gene_indices)]
        available = np.ones(source.shape, dtype=bool)
        if signature.coordinate_mask is not None:
            available &= signature.coordinate_mask[
                np.ix_(signature_target_indices, signature_gene_indices)
            ]
        destination = output[np.ix_(output_target_indices, output_gene_indices)]
        changed_coordinates += int(np.sum(available & (destination != source)))
        destination[available] = source[available]
        output[np.ix_(output_target_indices, output_gene_indices)] = destination
        local_write = write_mask[np.ix_(output_target_indices, output_gene_indices)]
        local_write[available] = True
        write_mask[np.ix_(output_target_indices, output_gene_indices)] = local_write
        written_by_context["context-independent"] = int(available.sum())
    else:
        assert fallback_contexts is not None
        fallback_context_lookup = {
            name: index for index, name in enumerate(fallback_contexts)
        }
        if signature.effects.ndim == 2:
            context_pairs = [(None, index) for index in range(len(fallback_contexts))]
        else:
            assert signature.contexts is not None
            context_pairs = [
                (signature_index, fallback_context_lookup[name])
                for signature_index, name in enumerate(signature.contexts)
                if name in fallback_context_lookup
                and (signature.context_mask is None or signature.context_mask[signature_index])
            ]
            require(context_pairs, "Signature has no available fallback context")
        for signature_context, output_context in context_pairs:
            source_matrix = (
                signature.effects
                if signature_context is None
                else signature.effects[signature_context]
            )
            source = source_matrix[np.ix_(signature_target_indices, signature_gene_indices)]
            available = np.ones(source.shape, dtype=bool)
            if signature.coordinate_mask is not None:
                source_mask = (
                    signature.coordinate_mask
                    if signature.coordinate_mask.ndim == 2
                    else signature.coordinate_mask[signature_context]
                )
                available &= source_mask[
                    np.ix_(signature_target_indices, signature_gene_indices)
                ]
            destination = output[output_context][
                np.ix_(output_target_indices, output_gene_indices)
            ]
            changed_coordinates += int(np.sum(available & (destination != source)))
            destination[available] = source[available]
            output[output_context][np.ix_(output_target_indices, output_gene_indices)] = destination
            local_write = write_mask[output_context][
                np.ix_(output_target_indices, output_gene_indices)
            ]
            local_write[available] = True
            write_mask[output_context][
                np.ix_(output_target_indices, output_gene_indices)
            ] = local_write
            written_by_context[fallback_contexts[output_context]] = int(available.sum())

    require(bool(write_mask.any()), "Signature availability masks remove every coordinate")
    target_gene_mask = np.any(write_mask, axis=0) if write_mask.ndim == 3 else write_mask
    statistics = {
        "signature_targets_on_challenge_axis": len(signature_target_indices),
        "signature_genes_on_challenge_axis_after_masks": len(signature_gene_indices),
        "signature_targets_outside_challenge": outside_targets,
        "signature_genes_outside_challenge": outside_genes,
        "overlaid_target_gene_coordinates": int(target_gene_mask.sum()),
        "written_coordinates": int(write_mask.sum()),
        "written_effect_values_including_contexts": int(write_mask.sum()),
        "numerically_changed_coordinates": changed_coordinates,
        "preserved_fallback_coordinates": int(output.size - write_mask.sum()),
        "preserved_fallback_effect_values_including_contexts": int(
            output.size - write_mask.sum()
        ),
        "written_coordinates_by_context": written_by_context,
    }
    return output, write_mask, statistics


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    for path in (
        args.bayesian_fallback,
        args.signature_prior,
        args.challenge_targets,
        args.challenge_genes,
    ):
        require(path.is_file(), f"Missing input: {path}")
    for output in (args.output_npz, args.output_json):
        require(not output.exists(), f"Refusing to overwrite {output}")

    challenge_targets = read_named_axis(
        args.challenge_targets,
        args.target_column,
        ("target", "targets", "gene"),
    )
    challenge_genes = read_named_axis(
        args.challenge_genes,
        args.gene_column,
        ("gene", "genes"),
    )
    fallback = load_effect_input(args.bayesian_fallback)
    signature = load_effect_input(args.signature_prior)
    if fallback.effects.ndim == 3:
        require(fallback.contexts is not None, "Contextual fallback lacks contexts")
    fallback_effects = align_fallback(fallback, challenge_targets, challenge_genes)
    overlaid, write_mask, statistics = overlay_signature(
        fallback_effects,
        fallback.contexts,
        challenge_targets,
        challenge_genes,
        signature,
    )
    require(np.isfinite(overlaid).all(), "Overlay produced non-finite effects")
    require(
        np.array_equal(overlaid[~write_mask], fallback_effects[~write_mask]),
        "Fallback changed outside the signature overlay mask",
    )

    for output in (args.output_npz, args.output_json):
        output.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {
        "effects": overlaid,
        "targets": np.asarray(challenge_targets, dtype="U"),
        "target_names": np.asarray(challenge_targets, dtype="U"),
        "genes": np.asarray(challenge_genes, dtype="U"),
        "gene_names": np.asarray(challenge_genes, dtype="U"),
        "target_cell_counts": np.zeros(len(challenge_targets), dtype=np.int64),
        "signature_overlay_mask": write_mask,
        "signature_overlay_target_mask": np.any(
            write_mask,
            axis=tuple(
                index
                for index in range(write_mask.ndim)
                if index != write_mask.ndim - 2
            ),
        ),
        "signature_overlay_gene_mask": np.any(
            write_mask, axis=tuple(range(write_mask.ndim - 1))
        ),
    }
    if fallback.contexts is not None:
        payload["contexts"] = np.asarray(fallback.contexts, dtype="U")
    np.savez_compressed(args.output_npz, **payload)

    report = {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "elapsed_seconds": time.time() - started,
        "inputs": {
            "bayesian_fallback": describe_file(args.bayesian_fallback),
            "bayesian_fallback_metadata": describe_companion_metadata(
                args.bayesian_fallback
            ),
            "signature_prior": describe_file(args.signature_prior),
            "signature_prior_metadata": describe_companion_metadata(
                args.signature_prior
            ),
            "challenge_targets": describe_file(args.challenge_targets),
            "challenge_genes": describe_file(args.challenge_genes),
        },
        "output": describe_file(args.output_npz),
        "axes": {
            "contexts": None if fallback.contexts is None else list(fallback.contexts),
            "targets": len(challenge_targets),
            "genes": len(challenge_genes),
            "effect_shape": list(overlaid.shape),
        },
        "overlay": statistics,
        "contract": {
            "fallback_coverage_required": "all-challenge-targets-and-genes",
            "signature_presence": "named-axis-intersection-and-explicit-effect-masks",
            "authoritative_signature_array": "effects",
            "effect_space_conversion": "none",
            "fallback_preserved_outside_overlay": True,
            "target_cell_counts": "zero-because-output-is-a-model-prior",
            "load_effect_atlas_compatible": True,
        },
        "runtime": {
            "implementation": describe_file(Path(__file__)),
            "argv": sys.argv,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "host": socket.gethostname(),
        },
    }
    args.output_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return report


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
