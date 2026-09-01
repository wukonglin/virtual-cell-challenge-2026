#!/usr/bin/env python3
"""Freeze a leakage-safe public panel for whole-context VCC validation.

Selection uses target names, treated-cell counts, gene-axis availability, and
ESM2 embeddings only. It never reads HepG2 or Jurkat perturbation effects. The
selected panel is balanced across deterministic ESM2 clusters, and complete
clusters are reserved as simulated unseen-target groups.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.cluster import KMeans

from generate_state_direct_counts import atomic_write_json
from infer_state_effect_prior import describe_file, read_single_column, require


SCHEMA = "vcc-public-validation-manifest-v1"
CONTROL_LABEL = "non-targeting"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k562-atlas",
        type=Path,
        default=Path("artifacts/v2/k562_essential_effect_atlas.npz"),
    )
    parser.add_argument(
        "--rpe1-atlas",
        type=Path,
        default=Path("artifacts/v2/rpe1_effect_atlas.npz"),
    )
    parser.add_argument(
        "--hepg2-raw",
        type=Path,
        default=Path(
            "dataset/raw/replogle_nadig/GSE264667_hepg2_raw_singlecell_01.h5ad"
        ),
    )
    parser.add_argument(
        "--jurkat-raw",
        type=Path,
        default=Path(
            "dataset/raw/replogle_nadig/GSE264667_jurkat_raw_singlecell_01.h5ad"
        ),
    )
    parser.add_argument(
        "--esm2",
        type=Path,
        default=Path("dataset/state_support/extracted/ESM2_pert_features.pt"),
    )
    parser.add_argument(
        "--state-genes",
        type=Path,
        default=Path("dataset/state_support/extracted/gene_names.csv"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("dataset/public_v51/split_manifest.csv"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("dataset/public_v51/split_manifest.json"),
    )
    parser.add_argument("--minimum-treated-cells", type=int, default=64)
    parser.add_argument("--panel-targets", type=int, default=300)
    parser.add_argument("--held-targets", type=int, default=33)
    parser.add_argument("--esm2-clusters", type=int, default=30)
    parser.add_argument("--expected-candidates", type=int, default=347)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args()


def _strings(values: np.ndarray) -> list[str]:
    return [
        value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def read_atlas_targets(path: Path) -> set[str]:
    """Read only the named target axis from a public atlas."""
    require(path.is_file(), f"Missing public atlas: {path}")
    with np.load(path, allow_pickle=False) as archive:
        require("target_names" in archive.files, f"Atlas lacks target_names: {path}")
        targets = _strings(archive["target_names"])
    require(len(targets) == len(set(targets)), f"Duplicate atlas targets: {path}")
    return set(targets)


def read_raw_metadata(path: Path) -> tuple[dict[str, int], set[str], dict[str, Any]]:
    """Read labels and gene names without accessing the expression matrix."""
    require(path.is_file(), f"Missing raw public data: {path}")
    data = ad.read_h5ad(path, backed="r")
    try:
        require("gene" in data.obs.columns, f"Missing gene labels in {path}")
        require("gene_name" in data.var.columns, f"Missing gene names in {path}")
        perturbations = data.obs["gene"].astype(str)
        counts = perturbations.value_counts(sort=False)
        target_counts = {
            str(target): int(count)
            for target, count in counts.items()
            if str(target) != CONTROL_LABEL
        }
        genes = set(data.var["gene_name"].astype(str).tolist())
        metadata = {
            "cells": int(data.n_obs),
            "input_gene_columns": int(data.n_vars),
            "unique_gene_symbols": int(len(genes)),
            "control_cells": int((perturbations == CONTROL_LABEL).sum()),
            "targets": int(len(target_counts)),
            "expression_matrix_accessed": False,
        }
    finally:
        data.file.close()
    return target_counts, genes, metadata


def load_esm2(path: Path) -> dict[str, np.ndarray]:
    require(path.is_file(), f"Missing ESM2 map: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    require(isinstance(payload, dict), "ESM2 input must be a target-to-tensor map")
    result = {
        str(name): np.asarray(value.detach().float().reshape(-1).numpy(), dtype=np.float32)
        for name, value in payload.items()
        if isinstance(value, torch.Tensor)
    }
    require(bool(result), "ESM2 input has no tensor values")
    dimensions = {int(value.size) for value in result.values()}
    require(len(dimensions) == 1, "ESM2 vectors have inconsistent dimensions")
    return result


def stable_target_key(target: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{target}".encode("utf-8")).hexdigest()


def balanced_panel_order(
    targets: list[str], labels: np.ndarray, *, seed: int
) -> list[str]:
    """Round-robin deterministic target order over embedding clusters."""
    cluster_labels = np.asarray(labels, dtype=np.int64)
    require(cluster_labels.shape == (len(targets),), "Cluster labels have bad shape")
    buckets: dict[int, list[str]] = {}
    for target, label in zip(targets, cluster_labels, strict=True):
        buckets.setdefault(int(label), []).append(target)
    for bucket in buckets.values():
        bucket.sort(key=lambda target: (stable_target_key(target, seed), target))
    result: list[str] = []
    offset = 0
    while len(result) < len(targets):
        added = 0
        for label in sorted(buckets):
            bucket = buckets[label]
            if offset < len(bucket):
                result.append(bucket[offset])
                added += 1
        require(added > 0, "Balanced cluster ordering stalled")
        offset += 1
    return result


def choose_holdout_clusters(
    cluster_counts: dict[int, int], desired_targets: int
) -> tuple[int, ...]:
    """Choose complete clusters with a target count closest to the request."""
    require(cluster_counts and desired_targets > 0, "Invalid holdout-cluster request")
    possibilities: dict[int, tuple[int, ...]] = {0: ()}
    for cluster, count in sorted(cluster_counts.items()):
        require(count > 0, "A selected cluster is empty")
        additions = {
            total + count: chosen + (cluster,)
            for total, chosen in possibilities.items()
        }
        for total, chosen in additions.items():
            previous = possibilities.get(total)
            if previous is None or chosen < previous:
                possibilities[total] = chosen
    positive = [(total, chosen) for total, chosen in possibilities.items() if total > 0]
    require(bool(positive), "No non-empty holdout cluster set exists")
    _, selected = min(
        positive,
        key=lambda item: (
            abs(item[0] - desired_targets),
            item[0] > desired_targets,
            len(item[1]),
            item[1],
        ),
    )
    return selected


def atomic_write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        frame.to_csv(handle, index=False)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    require(args.minimum_treated_cells > 0, "minimum-treated-cells must be positive")
    require(args.panel_targets > 0, "panel-targets must be positive")
    require(0 < args.held_targets < args.panel_targets, "Invalid held-target count")
    require(args.esm2_clusters > 1, "At least two ESM2 clusters are required")
    require(args.expected_candidates >= args.panel_targets, "Invalid candidate expectation")
    require(args.output_csv.suffix == ".csv", "output-csv must end in .csv")
    require(args.output_json.suffix == ".json", "output-json must end in .json")
    for output in (args.output_csv, args.output_json):
        require(not output.exists(), f"Refusing to overwrite frozen manifest: {output}")

    k562 = read_atlas_targets(args.k562_atlas.resolve())
    rpe1 = read_atlas_targets(args.rpe1_atlas.resolve())
    hepg2_counts, hepg2_genes, hepg2_metadata = read_raw_metadata(
        args.hepg2_raw.resolve()
    )
    jurkat_counts, jurkat_genes, jurkat_metadata = read_raw_metadata(
        args.jurkat_raw.resolve()
    )
    state_genes = set(read_single_column(args.state_genes.resolve()))
    esm2 = load_esm2(args.esm2.resolve())

    stages: dict[str, set[str]] = {}
    stages["k562_and_rpe1"] = k562 & rpe1
    stages["with_esm2"] = stages["k562_and_rpe1"] & set(esm2)
    stages["hepg2_min_cells"] = {
        target
        for target in stages["with_esm2"]
        if hepg2_counts.get(target, 0) >= args.minimum_treated_cells
    }
    stages["jurkat_min_cells"] = {
        target
        for target in stages["hepg2_min_cells"]
        if jurkat_counts.get(target, 0) >= args.minimum_treated_cells
    }
    stages["target_on_both_public_gene_axes"] = {
        target
        for target in stages["jurkat_min_cells"]
        if target in hepg2_genes and target in jurkat_genes
    }
    stages["target_on_state_support_axis"] = (
        stages["target_on_both_public_gene_axes"] & state_genes
    )
    candidates = sorted(stages["target_on_state_support_axis"])
    require(
        len(candidates) == args.expected_candidates,
        f"Expected {args.expected_candidates} candidates, found {len(candidates)}",
    )

    features = np.stack([esm2[target] for target in candidates]).astype(np.float32)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    require(np.all(np.isfinite(norms)) and np.all(norms > 0), "Invalid ESM2 norms")
    features /= norms
    clustering = KMeans(
        n_clusters=args.esm2_clusters,
        random_state=args.seed,
        n_init=20,
        algorithm="lloyd",
    )
    labels = clustering.fit_predict(features).astype(np.int64)
    require(len(np.unique(labels)) == args.esm2_clusters, "An ESM2 cluster is empty")
    label_by_target = dict(zip(candidates, labels.tolist(), strict=True))
    ordered = balanced_panel_order(candidates, labels, seed=args.seed)
    selected = ordered[: args.panel_targets]
    selected_cluster_counts = {
        int(cluster): int(sum(label_by_target[target] == cluster for target in selected))
        for cluster in sorted(set(labels.tolist()))
        if any(label_by_target[target] == cluster for target in selected)
    }
    held_clusters = choose_holdout_clusters(
        selected_cluster_counts, args.held_targets
    )
    held_set = {
        target for target in selected if label_by_target[target] in held_clusters
    }
    require(
        len(held_set) == args.held_targets,
        f"Complete held clusters contain {len(held_set)}, not {args.held_targets}, targets",
    )

    frame = pd.DataFrame(
        {
            "target_gene": selected,
            "panel_ordinal": np.arange(len(selected), dtype=np.int64),
            "esm2_cluster": [label_by_target[target] for target in selected],
            "validation_route": [
                "held_target" if target in held_set else "direct" for target in selected
            ],
            "hepg2_treated_cells": [hepg2_counts[target] for target in selected],
            "jurkat_treated_cells": [jurkat_counts[target] for target in selected],
        }
    )
    require(frame["target_gene"].is_unique, "Selected panel contains duplicate targets")
    direct_targets = args.panel_targets - args.held_targets
    require(
        int((frame["validation_route"] == "direct").sum()) == direct_targets,
        "Bad direct count",
    )
    require(
        int((frame["validation_route"] == "held_target").sum()) == args.held_targets,
        "Bad held-target count",
    )

    atomic_write_csv(args.output_csv, frame)
    report = {
        "schema": SCHEMA,
        "selection_contract": {
            "effect_values_accessed": False,
            "allowed_inputs": [
                "public target names",
                "public treated-cell counts",
                "public gene-axis membership",
                "STATE support-axis membership",
                "ESM2 target embeddings",
            ],
            "minimum_treated_cells_per_context": args.minimum_treated_cells,
            "panel_targets": args.panel_targets,
            "direct_targets": direct_targets,
            "held_targets": args.held_targets,
            "held_clusters": list(held_clusters),
            "esm2_clusters": args.esm2_clusters,
            "seed": args.seed,
            "cluster_algorithm": "sklearn.KMeans/lloyd/l2-normalized-ESM2",
            "panel_selection": "seeded-hash-within-cluster-round-robin",
            "held_selection": "complete-cluster-subset-nearest-requested-count",
        },
        "filter_counts": {
            name: len(values) for name, values in stages.items()
        },
        "selected_cluster_counts": {
            str(cluster): count for cluster, count in selected_cluster_counts.items()
        },
        "public_metadata": {
            "hepg2": hepg2_metadata,
            "jurkat": jurkat_metadata,
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
        "provenance": {
            "k562_atlas": describe_file(args.k562_atlas.resolve()),
            "rpe1_atlas": describe_file(args.rpe1_atlas.resolve()),
            "hepg2_raw": describe_file(args.hepg2_raw.resolve()),
            "jurkat_raw": describe_file(args.jurkat_raw.resolve()),
            "esm2": describe_file(args.esm2.resolve()),
            "state_genes": describe_file(args.state_genes.resolve()),
            "output_csv": describe_file(args.output_csv.resolve()),
        },
    }
    atomic_write_json(args.output_json, report)
    print(f"Wrote frozen public panel: {args.output_csv}")
    print(f"Wrote manifest provenance: {args.output_json}")


if __name__ == "__main__":
    main()
