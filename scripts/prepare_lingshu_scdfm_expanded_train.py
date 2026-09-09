#!/usr/bin/env python3
"""Expand only public K562 training rows; freeze parent validation and gene axes.

This does not read RPE1 or challenge expression, select genes, or alter quality
thresholds. Original selected treated rows are retained, then additional rows
are sampled without replacement. Controls remain random gem-group-matched
public cells, not observed treated/control pairs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import anndata as ad
import numpy as np

try:
    from .prepare_lingshu_scdfm_data import (
        CONTROL, _validate_source, paired_controls, read_axis, read_selected,
        sha256_file, stable_rng,
    )
except ImportError:
    from prepare_lingshu_scdfm_data import (
        CONTROL, _validate_source, paired_controls, read_axis, read_selected,
        sha256_file, stable_rng,
    )


SCHEMA = "vcc-lingshu-scdfm-expanded-train-public-cache-v3"
PARENT_SCHEMA = "vcc-lingshu-scdfm-public-cache-v1"
AXES = ("gene_names", "gene_indices", "normalization_gene_names")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def array_digest(value):
    """Bind dtype and shape as well as exact C-order array bytes."""
    digest = hashlib.sha256()
    digest.update(json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True).encode())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def expanded_rows(labels, targets, parent_rows, maximum, seed):
    require(parent_rows.dtype.kind in "iu" and parent_rows.ndim == 1, "Invalid parent treated row IDs")
    require(len(set(parent_rows.tolist())) == len(parent_rows), "Duplicate parent treated rows")
    require(bool(len(parent_rows)) and parent_rows.min() >= 0 and parent_rows.max() < len(labels), "Parent row IDs out of bounds")
    require(set(labels[parent_rows]) == set(targets), "Parent treated row target identities differ")
    result = []
    for target in sorted(targets):
        old = parent_rows[labels[parent_rows] == target]
        require(len(old) <= maximum, "New training cap would discard parent rows")
        available = np.flatnonzero(labels == target)
        remaining = np.setdiff1d(available, old, assume_unique=True)
        count = min(maximum - len(old), len(remaining))
        extra = stable_rng(seed, "expanded-train-v3", "K562", target).choice(remaining, count, replace=False)
        result.append(np.sort(np.concatenate((old, extra))))
    return np.concatenate(result).astype(np.int64)


def split_half_diagnostic(x, control, targets, seed):
    """Training-only split-half empirical mean-effect discrepancy.

Controls can recur across rows, so this is descriptive stability, not an
independent variance estimate, significance test, or model-selection metric.
"""
    groups = []
    for target in sorted(set(targets.tolist())):
        rows = np.flatnonzero(targets == target)
        require(len(rows) >= 2, "Split-half diagnostics need at least two cells per target")
        permuted = stable_rng(seed, "split-half-group-effect", target).permutation(rows)
        halves = np.array_split(permuted, 2)
        effects = [x[part].astype(np.float64).mean(axis=0) - control[part].astype(np.float64).mean(axis=0)
                   for part in halves]
        mse = float(np.mean(np.square(effects[0] - effects[1])))
        require(np.isfinite(mse), "Nonfinite training split-half discrepancy")
        groups.append({"target": target, "cells": len(rows), "half_cells": [len(part) for part in halves],
                       "split_half_group_effect_mse": mse})
    return {"equal_target_mean_mse": float(np.mean([item["split_half_group_effect_mse"] for item in groups])),
            "target_count": len(groups), "groups": groups}


def verify_existing(base_cache, output_npz, output_json):
    """Read-only portable verification; never open full source expression."""
    receipt = json.loads(output_json.read_text())
    require(receipt.get("schema") == SCHEMA and receipt.get("challenge_treated_used") is False, "Wrong expansion receipt scope")
    parent_sha, output_sha = sha256_file(base_cache), sha256_file(output_npz)
    require(receipt["expansion"]["parent_cache"]["sha256"] == parent_sha, "Parent cache hash mismatch")
    require(receipt["artifacts"]["cache"]["sha256"] == output_sha, "Expanded cache hash mismatch")
    with np.load(base_cache, allow_pickle=False) as archive:
        parent = {key: archive[key] for key in archive.files}
    with np.load(output_npz, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    old = json.loads(parent["metadata_json"].item())
    require(old.get("schema") == PARENT_SCHEMA, "Unsupported parent cache schema")
    metadata = json.loads(arrays["metadata_json"].item())
    require(metadata == {key: value for key, value in receipt.items() if key != "artifacts"}, "Embedded expansion metadata differs from receipt")
    require(old.get("challenge_treated_used") is False and metadata["seed"] == old["seed"], "Parent scope or seed differs")
    frozen_names = sorted(set(AXES) | {name for name in parent if name.startswith("val_")})
    frozen_hashes = {name: array_digest(parent[name]) for name in frozen_names}
    require(metadata["expansion"]["frozen_arrays_sha256"] == frozen_hashes, "Frozen-array receipt differs from parent")
    for name in frozen_names:
        require(array_digest(arrays[name]) == frozen_hashes[name], f"Frozen parent array changed: {name}")
    require(set(arrays) == set(parent), "Cache fields differ from parent")
    targets = sorted(set(parent["train_targets"].tolist()))
    require(sorted(set(arrays["train_targets"].tolist())) == targets == metadata["split"]["training_targets"], "Training target identities changed")
    require(metadata["expansion"]["training_target_count"] == len(targets), "Training target count differs")
    require(set(arrays["train_contexts"].tolist()) == {"K562"}, "Expanded training contains another context")
    require(metadata["normalization"] == old["normalization"], "Normalization metadata changed")
    require(metadata["sources"] == old["sources"] and metadata["expansion"]["K562_source_sha256_verified_against_parent"] == old["sources"]["K562"]["sha256"], "Source identity differs")
    require(metadata["split"]["heldout_targets"] == old["split"]["heldout_targets"], "Heldout target split changed")
    require(not set(targets).intersection(old["split"]["heldout_targets"]), "Training includes heldout targets")
    rows, control_rows = arrays["train_source_rows"], arrays["train_control_source_rows"]
    count, gene_count = len(rows), len(arrays["gene_names"])
    require(rows.dtype.kind in "iu" and rows.ndim == 1 and len(set(rows.tolist())) == count, "Invalid or duplicate expanded treated rows")
    require(set(parent["train_source_rows"].tolist()) <= set(rows.tolist()), "Original treated rows were not retained")
    require(count == metadata["split"]["train_rows"] == metadata["expansion"]["expanded_train_rows"], "Expanded row count mismatch")
    require(metadata["expansion"]["parent_train_rows"] == len(parent["train_x"]), "Parent row count mismatch")
    for name in ("train_x", "train_control"):
        value = arrays[name]
        require(value.shape == (count, gene_count) and value.dtype == np.float32, "Invalid expanded expression shape/dtype")
        require(np.isfinite(value).all() and (value >= 0).all() and (value <= np.log1p(10000) + 0.01).all(), "Invalid expanded log-CP10k values")
    for name in ("train_targets", "train_contexts", "train_control_source_rows"):
        require(arrays[name].shape == (count,), "Expanded row-aligned array shape mismatch")
    require(control_rows.dtype.kind in "iu", "Invalid expanded control row dtype")
    for value in (rows, control_rows):
        require(value.min() >= 0 and value.max() < old["sources"]["K562"]["shape"][0], "Expanded source row out of bounds")
    lookup = {int(row): str(target) for row, target in zip(rows, arrays["train_targets"], strict=True)}
    require(all(lookup[int(row)] == str(target) for row, target in zip(parent["train_source_rows"], parent["train_targets"], strict=True)), "Retained treated-row labels changed")
    require(metadata["expansion"]["selected_train_rows_sha256"] == array_digest(rows) and
            metadata["expansion"]["selected_train_control_rows_sha256"] == array_digest(control_rows), "Selected source-row digest mismatch")
    per_target = {target: int(np.sum(arrays["train_targets"] == target)) for target in targets}
    require(per_target == metadata["expansion"]["per_target_cells"] and
            max(per_target.values()) <= metadata["expansion"]["new_train_cells_per_target_cap"], "Per-target cell cap/count differs")
    return {"status": "pass", "schema": SCHEMA, "parent_cache_sha256": parent_sha,
            "expanded_cache_sha256": output_sha, "train_rows": count, "training_targets": len(targets),
            "frozen_array_count": len(frozen_names), "all_frozen_arrays_bit_identical": True,
            "all_parent_treated_rows_retained": True, "source_expression_read": False}


def prepare(args):
    outputs = (args.output_npz, args.output_json)
    require(outputs[0].resolve() != outputs[1].resolve(), "Outputs must have different paths")
    inputs = {path.resolve() for path in (args.base_cache, args.k562, args.public_genes)}
    for output in outputs:
        require(output.resolve() not in inputs, "Output must not replace an input")
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
    require(min(args.max_train_cells, args.max_controls_per_batch, args.block_rows) > 0, "Cell caps and block size must be positive")
    parent_sha = sha256_file(args.base_cache)
    with np.load(args.base_cache, allow_pickle=False) as archive:
        parent = {key: archive[key] for key in archive.files}
    metadata = json.loads(parent["metadata_json"].item())
    require(metadata.get("schema") == PARENT_SCHEMA, "Unsupported parent cache schema")
    require(metadata.get("challenge_treated_used") is False, "Parent used challenge-treated expression")
    require(metadata.get("seed") == args.seed, "Seed must match the frozen parent split")
    require(metadata["normalization"].get("full_axis_scope") == "shared_public_challenge_gene_axis" and
            metadata["normalization"].get("target_sum") == 10000 and
            metadata["normalization"].get("space") == "log1p_library_normalized_full_axis", "Unsupported parent normalization")
    require(set(parent["train_contexts"].tolist()) == {"K562"}, "Parent training must be K562 only")
    targets = sorted(set(parent["train_targets"].tolist()))
    require(targets == metadata["split"]["training_targets"], "Parent target identities disagree with split metadata")
    heldout = set(metadata["split"]["heldout_targets"])
    require(not heldout.intersection(targets), "Heldout targets leak into parent training")
    require(set(parent["val_targets"][parent["val_kind"] == "target"].tolist()) == heldout, "Parent whole-target holdout differs")
    require(set(parent["val_kind"].tolist()) == {"context", "target"}, "Both parent validation kinds required")
    require(set(parent["val_contexts"][parent["val_kind"] == "context"].tolist()) == {"RPE1"}, "Parent context holdout differs")
    frozen_names = sorted(set(AXES) | {name for name in parent if name.startswith("val_")})
    frozen_digests = {name: array_digest(parent[name]) for name in frozen_names}
    expected_source = metadata["sources"]["K562"]
    source_sha = sha256_file(args.k562)
    require(source_sha == expected_source["sha256"], "K562 source hash differs from authenticated parent source")
    require(args.k562.stat().st_size == expected_source["size_bytes"], "K562 source size differs")
    public_genes = read_axis(args.public_genes, "gene_name")
    source = ad.read_h5ad(args.k562, backed="r")
    try:
        labels, batches, genes = _validate_source(source, "k562")
        require(list(source.shape) == expected_source["shape"], "K562 source shape differs")
        require(genes == public_genes, "Public gene CSV differs from authenticated K562 axis")
        require(set(labels) - {CONTROL} == set(targets) | heldout, "Public source target panel differs from parent split")
        require(np.array_equal(labels[parent["train_source_rows"]], parent["train_targets"]), "Parent training labels do not match source rows")
        require(np.all(labels[parent["train_control_source_rows"]] == CONTROL), "Parent training control rows are not controls")
        lookup = {name: index for index, name in enumerate(genes)}
        require(set(parent["normalization_gene_names"]) <= set(lookup), "Normalization genes missing from source")
        require(set(parent["gene_names"]) <= set(parent["normalization_gene_names"]), "Modeled genes outside fixed normalization axis")
        norm_columns = np.asarray([lookup[name] for name in parent["normalization_gene_names"]], dtype=np.int64)
        columns = np.asarray([lookup[name] for name in parent["gene_names"]], dtype=np.int64)
        rows = expanded_rows(labels, targets, parent["train_source_rows"], args.max_train_cells, args.seed)
        require(not heldout.intersection(labels[rows]), "Expanded rows include heldout targets")
        control_rows, matching = paired_controls(labels, batches, rows, args.max_controls_per_batch, args.seed, "K562")
        arrays = {name: value.copy() for name, value in parent.items()}
        arrays.update(train_x=read_selected(source, rows, columns, args.block_rows, norm_columns),
                      train_control=read_selected(source, control_rows, columns, args.block_rows, norm_columns),
                      train_targets=np.asarray(labels[rows], dtype=parent["train_targets"].dtype),
                      train_contexts=np.full(len(rows), "K562", dtype=parent["train_contexts"].dtype),
                      train_source_rows=rows, train_control_source_rows=control_rows)
    finally:
        source.file.close()
    for name in frozen_names:
        require(array_digest(arrays[name]) == frozen_digests[name], f"Frozen parent array changed: {name}")
    require(set(arrays["train_targets"].tolist()) == set(targets), "Training target panel changed")
    require(sha256_file(args.base_cache) == parent_sha, "Parent cache changed while preparing expanded rows")
    diagnostic = {
        "scope": "training_only_descriptive_split_half_group_mean_effect_stability",
        "independent_test": False, "used_for_selection_or_threshold_tuning": False,
        "caveat": "Random matched controls may recur across rows; not an independent variance estimate.",
        "parent": split_half_diagnostic(parent["train_x"], parent["train_control"], parent["train_targets"], args.seed),
        "expanded": split_half_diagnostic(arrays["train_x"], arrays["train_control"], arrays["train_targets"], args.seed),
    }
    new_metadata = copy.deepcopy(metadata)
    new_metadata.update(schema=SCHEMA, script_sha256=sha256_file(Path(__file__)), training_gene_selection=False)
    new_metadata["gene_selection"].update(inherited_unchanged_from_parent=True, reranked_during_expansion=False)
    new_metadata["split"].pop("max_cells_per_target", None)
    new_metadata["split"].update(train_rows=len(rows), max_train_cells_per_target=args.max_train_cells,
                                   validation_max_cells_per_target=metadata["split"].get("max_cells_per_target"))
    new_metadata["control_matching"]["train"] = matching
    new_metadata["expansion"] = {
        "parent_cache": {"path": str(args.base_cache.resolve()), "sha256": parent_sha},
        "parent_schema": metadata["schema"], "parent_train_rows": len(parent["train_x"]),
        "expanded_train_rows": len(rows), "training_target_count": len(targets),
        "sampling": "retain_all_parent_treated_rows_then_uniform_without_replacement_from_remaining_per_target",
        "parent_treated_rows_retained": True, "new_train_cells_per_target_cap": args.max_train_cells,
        "new_control_pool_cells_per_gem_cap": args.max_controls_per_batch,
        "training_targets_unchanged": True, "validation_arrays_bit_identical": True,
        "normalization_and_selected_gene_arrays_bit_identical": True,
        "no_gene_reselection": True, "RPE1_expression_reread": False, "challenge_expression_read": False,
        "frozen_arrays_sha256": frozen_digests,
        "selected_train_rows_sha256": array_digest(rows),
        "selected_train_control_rows_sha256": array_digest(control_rows),
        "public_gene_axis_sha256": sha256_file(args.public_genes),
        "K562_source_sha256_verified_against_parent": source_sha,
        "source_reader_sha256": sha256_file(Path(__file__).with_name("prepare_lingshu_scdfm_data.py")),
        "per_target_cells": {target: int(np.sum(arrays["train_targets"] == target)) for target in targets},
        "training_only_noise_diagnostic": diagnostic,
    }
    arrays["metadata_json"] = np.asarray(json.dumps(new_metadata, sort_keys=True, allow_nan=False), dtype=str)
    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
    with args.output_npz.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    receipt = copy.deepcopy(new_metadata)
    receipt["artifacts"] = {"cache": {"path": str(args.output_npz.resolve()), "sha256": sha256_file(args.output_npz)}}
    with args.output_json.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return receipt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-cache", type=Path, default=Path("artifacts/lingshu_scdfm/public_cache_v1.npz"))
    parser.add_argument("--k562", type=Path, default=Path("dataset/state_support/extracted/k562_gwps.h5"))
    parser.add_argument("--public-genes", type=Path, default=Path("dataset/state_support/extracted/gene_names.csv"))
    parser.add_argument("--output-npz", type=Path, default=Path("artifacts/lingshu_scdfm/public_cache_expanded_train_v3.npz"))
    parser.add_argument("--output-json", type=Path, default=Path("artifacts/lingshu_scdfm/public_cache_expanded_train_v3.json"))
    parser.add_argument("--max-train-cells", type=int, default=512)
    parser.add_argument("--max-controls-per-batch", type=int, default=512)
    parser.add_argument("--block-rows", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260909)
    parser.add_argument("--verify-existing", action="store_true", help="Verify existing cache/receipt against parent without reading K562, RPE1, or challenge expression")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.verify_existing:
        print(json.dumps(verify_existing(args.base_cache, args.output_npz, args.output_json), indent=2))
    else:
        receipt = prepare(args)
        diagnostic = receipt["expansion"]["training_only_noise_diagnostic"]
        print(json.dumps({"schema": receipt["schema"], "train_rows": receipt["split"]["train_rows"],
                          "training_targets": receipt["expansion"]["training_target_count"],
                          "parent_split_half_mse": diagnostic["parent"]["equal_target_mean_mse"],
                          "expanded_split_half_mse": diagnostic["expanded"]["equal_target_mean_mse"],
                          "artifacts": receipt["artifacts"]}, indent=2))
