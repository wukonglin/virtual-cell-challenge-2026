"""Pure, fixed-partition effect reliability diagnostics; no I/O or fitting.

The caller must authenticate the v2 cache and preregister this implementation.
The former context NTC pool is reassigned to pseudo-treated controls ONLY for
this model-free audit. Thirty-two seeded partitions reuse the same cells and
are not biological replicates. No p-values, confidence intervals, truncation
of negative crossdots, fitted thresholds, or model outputs are used.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
from collections import defaultdict
import numpy as np

SCHEMA = "public-effect-reliability-v1"
SEEDS = tuple(range(20260920, 20260952))
AGGREGATE_KEYS = {
    "real_group_energy": "observed_group_effect_mse", "null_group_energy": "null_group_effect_mse",
    "real_target_energy": "observed_target_pooled_effect_mse", "null_target_energy": "null_target_pooled_effect_mse",
    "real_half_crossdot": "observed_within_batch_half_crossdot", "null_half_crossdot": "null_within_batch_half_crossdot",
    "real_half_agreement": "observed_within_batch_half_agreement", "null_half_agreement": "null_within_batch_half_agreement",
    "real_between_batch_crossdot": "observed_between_batch_crossdot", "null_between_batch_crossdot": "null_between_batch_crossdot",
    "real_half_disagreement_mse": "observed_within_batch_half_disagreement_mse", "null_half_disagreement_mse": "null_within_batch_half_disagreement_mse",
}


def policy():
    return {
        "schema": SCHEMA, "seeds": list(SEEDS), "partitions": 32,
        "scope": "existing authenticated K562/Jurkat v2 calibration cache; descriptive audit only",
        "role_reassignment": "context NTCs become pseudo-treated controls in a model-free audit; no context encoder/model fitted",
        "actual_effect": "treated n-cell mean minus all32 anchor/reference mean; n exactly8..32 per group",
        "matched_null": "prefix n of source/batch-global context32 permutation minus all32 anchor mean",
        "rng": "PCG64 via SHA256 of compact JSON[seed,namespace,source,batch,target_or_null]; rows first sorted by source-row ID",
        "pool_permutations": "one context and one independent anchor permutation per source/batch/seed, reused across targets",
        "halves": "treated floor(n/2),ceil(n/2); context null uses corresponding disjoint prefix halves; anchor16+16 disjoint halves",
        "group_energy": "mean squared genes of full treated-minus-anchor mean; same operation for null",
        "target_pooled_energy": "equal-batch mean of delta vectors FIRST, then mean squared genes; matching null operation",
        "within_batch_half_crossdot": "mean gene product of independent-half deltas; keep negative values",
        "between_batch_crossdot": "equal unordered distinct-batch pairs within target/source, mean gene delta product; keep negatives",
        "half_agreement": "2*dot/(energyA+energyB); null if denominator exactly0; descriptive agreement, not ICC/Pearson",
        "half_disagreement": "mean squared gene difference between independently anchored halves",
        "weighting": "equal batches within target; equal targets within source; equal sources overall",
        "undefined_values": "omit undefined agreement values in descriptive means and report defined counts; never zero-fill",
        "partition_summaries": "mean,min,max,population SD across32 resamples of fixed cells; ranges are NOT confidence intervals",
        "null_full32": "when n=32 the full null mean is invariant across partitions, up to arithmetic precision",
        "sham_pool": "eight sham NTCs not used in statistics; the caller's authenticated cache loader may integrity-validate every pool",
        "limitations": ["same-target public source transfer, not unseen-target evaluation",
                       "controls reused across target groups; groups/partitions are not independent replicates",
                       "matched-null energy excess also includes treated/control variance differences divided by n",
                       "crossdots are descriptive reproducibility diagnostics, not causal effect proofs",
                       "public sources and v2 cells already exposed in this project; no untouched-test claim"],
        "fitting": False, "model_outputs_read": False, "pvalues": False,
        "confidence_intervals": False, "threshold_selection": False, "negative_truncation": False,
    }


@dataclass(frozen=True)
class Group:
    index: int
    source: str
    target: str
    batch: str
    treated: np.ndarray
    context: np.ndarray
    anchor: np.ndarray


def _strings(value, name, *, unique=False):
    value = np.asarray(value)
    if value.ndim != 1 or value.dtype.kind != "U" or len(value) == 0:
        raise ValueError(f"{name} must be a nonempty Unicode axis")
    result = tuple(str(x) for x in value)
    if any(not x or x.strip() != x for x in result) or unique and len(set(result)) != len(result):
        raise ValueError(f"Malformed {name} axis")
    return result


def _extract(arrays):
    """Validate used arrays and globally consistent pools; ignore sham values."""
    genes = _strings(arrays["genes"], "genes", unique=True)
    targets, sources, batches = (_strings(arrays[key], key) for key in ("targets", "sources", "batches"))
    k, width = len(targets), len(genes)
    if not 1 <= width <= 512 or not 1 <= k <= 448 or len(sources) != k or len(batches) != k:
        raise ValueError("Reliability audit exceeds the bounded v2 axes")
    if set(sources) != {"replogle_k562", "nadig_jurkat"}:
        raise ValueError("Only the two authenticated v2 calibration sources are admitted")
    identities = tuple(zip(sources, targets, batches))
    if len(set(identities)) != k:
        raise ValueError("Repeated source/target/batch group")
    memberships, matrices = {}, {}
    for name in ("treated", "context_control", "control"):
        matrix, rows = np.asarray(arrays[name]), np.asarray(arrays[name + "_rows"])
        labels = np.asarray(arrays["group" if name == "treated" else name + "_group"])
        if (matrix.ndim != 2 or matrix.shape[1] != width or matrix.dtype.kind != "f"
                or not np.isfinite(matrix).all() or np.any(matrix < 0)):
            raise ValueError("Audit values must be finite nonnegative expression")
        if (rows.dtype.kind not in "iu" or labels.dtype.kind not in "iu" or rows.shape != (len(matrix),)
                or labels.shape != (len(matrix),) or np.any(rows < 0) or np.any(labels < 0) or np.any(labels >= k)):
            raise ValueError("Malformed row IDs or group assignments")
        counts = np.bincount(labels.astype(np.int64), minlength=k)
        if (name == "treated" and (np.any(counts < 8) or np.any(counts > 32))) or name != "treated" and np.any(counts != 32):
            raise ValueError("Each group needs8..32treated and32context+32anchor controls")
        memberships[name] = (rows, labels)
        matrices[name] = matrix
    pools, all_roles, unique_rows = {}, {}, defaultdict(set)
    groups = []
    for index, (source, target, batch) in enumerate(identities):
        extracted = {}
        for name in ("treated", "context_control", "control"):
            rows, labels = memberships[name]
            indices = np.flatnonzero(labels == index)
            order = np.argsort(rows[indices], kind="stable")
            ids = tuple(int(x) for x in rows[indices][order])
            if len(set(ids)) != len(ids):
                raise ValueError("Duplicate cells inside a pool")
            values = matrices[name][indices][order].astype(np.float64)
            for row in ids:
                identity = (source, row)
                role = (name, batch, target if name == "treated" else None)
                if identity in all_roles and (all_roles[identity] != role or name == "treated"):
                    raise ValueError("Control roles/batches overlap or treated cells repeat")
                all_roles[identity] = role
                unique_rows[source, name].add(row)
            key = (source, batch, name)
            if name != "treated":
                if key in pools:
                    old_ids, old_values = pools[key]
                    if ids != old_ids or not np.array_equal(values, old_values):
                        raise ValueError("NTC pools must match globally within source/batch")
                    values = old_values
                else:
                    pools[key] = (ids, values)
            extracted[name] = values
        groups.append(Group(index, source, target, batch, extracted["treated"], extracted["context_control"], extracted["control"]))
    per_source = {}
    for source in sorted(set(sources)):
        selected = [g for g in groups if g.source == source]
        target_counts = {target: sum(g.target == target for g in selected) for target in {g.target for g in selected}}
        if any(not 2 <= count <= 4 for count in target_counts.values()):
            raise ValueError("Every target requires two to four matching batches per source")
        per_source[source] = {"groups": len(selected), "targets": len(target_counts), "source_batches": len({g.batch for g in selected}),
                              **{name + "_unique_cells": len(unique_rows[source, name]) for name in matrices},
                              **{name + "_entries": sum(len(getattr(g, {"treated": "treated", "control": "anchor", "context_control": "context"}[name])) for g in selected) for name in matrices}}
    if set(g.target for g in groups if g.source == "replogle_k562") != set(g.target for g in groups if g.source == "nadig_jurkat"):
        raise ValueError("Both calibration sources must cover identical targets")
    counts = {"genes": width, "groups": k, "sources": 2, "common_targets": len(set(targets)),
              "source_target_pairs": sum(row["targets"] for row in per_source.values()), "per_source": per_source,
              "source_batches": sum(row["source_batches"] for row in per_source.values()), "sham_cells_used": 0}
    for name in matrices:
        counts[name + "_unique_cells"] = sum(row[name + "_unique_cells"] for row in per_source.values())
        counts[name + "_entries"] = sum(row[name + "_entries"] for row in per_source.values())
    return groups, counts


def _permutation(size, seed, namespace, source, batch, target=None):
    encoded = json.dumps([seed, namespace, source, batch, target], separators=(",", ":"), ensure_ascii=True).encode()
    entropy = int.from_bytes(hashlib.sha256(encoded).digest(), "big")
    return np.random.Generator(np.random.PCG64(entropy)).permutation(size)


def _mean(values):
    defined = [float(value) for value in values if value is not None]
    return float(np.mean(defined)) if defined else None


def _energy(vector):
    return float(np.mean(np.square(vector)))


def half_statistics(first, second):
    first, second = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if first.ndim != 1 or first.shape != second.shape or not len(first) or not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("Half deltas must be aligned finite gene vectors")
    crossdot = float(np.mean(first * second))
    denominator = _energy(first) + _energy(second)
    return {"crossdot": crossdot, "agreement": float(2 * crossdot / denominator) if denominator != 0 else None,
            "disagreement_mse": _energy(first - second)}


def between_batch_crossdot(vectors):
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.ndim != 2 or not 2 <= len(matrix) <= 4 or matrix.shape[1] < 1 or not np.isfinite(matrix).all():
        raise ValueError("Between-batch crossdot needs two to four finite aligned batch vectors")
    return float(np.mean([np.mean(matrix[i] * matrix[j]) for i in range(len(matrix)) for j in range(i)]))


def summarize_partitions(values):
    values = tuple(values)
    defined = np.array([float(value) for value in values if value is not None], dtype=np.float64)
    if not np.isfinite(defined).all():
        raise ValueError("Nonfinite partition statistic")
    return {"mean": float(defined.mean()) if len(defined) else None,
            "min": float(defined.min()) if len(defined) else None,
            "max": float(defined.max()) if len(defined) else None,
            "partition_sd": float(defined.std()) if len(defined) else None,
            "defined_partitions": len(defined), "partitions": len(values), "range_is_confidence_interval": False}


def _summarize_rows(rows):
    keys = set(rows[0])
    if any(set(row) != keys for row in rows):
        raise ValueError("Partition metric roster changed")
    return {key: summarize_partitions(row[key] for row in rows) for key in sorted(keys)}


def _mean_rows(rows):
    return {key: _mean(row[key] for row in rows) for key in sorted(rows[0])}


def compute_reliability(arrays, *, seeds=SEEDS):
    """Return descriptive group/target/source statistics; mutates no input array."""
    seeds = tuple(seeds)
    if seeds != SEEDS or any(type(seed) is not int for seed in seeds):
        raise ValueError("Exactly the32 preregistered seeds are required; no adaptive repetition")
    groups, input_counts = _extract(arrays)
    observed = [g.treated.mean(0) - g.anchor.mean(0) for g in groups]
    cohorts = sorted({(g.source, g.target) for g in groups})
    target_groups = {key: [i for i, g in enumerate(groups) if (g.source, g.target) == key] for key in cohorts}
    batch_groups = {(g.source, g.batch): g for g in groups}
    group_history = [[] for _ in groups]
    target_history = {key: [] for key in cohorts}
    per_seed = []
    for seed in seeds:
        permutations = {key: (
            _permutation(32, seed, "context", *key), _permutation(32, seed, "anchor", *key)
        ) for key in sorted(batch_groups)}
        group_metrics, null_vectors, real_halves, null_halves = [], [], [], []
        for index, group in enumerate(groups):
            context_order, anchor_order = permutations[group.source, group.batch]
            n = len(group.treated); n1 = n // 2
            treated_order = _permutation(n, seed, "treated", group.source, group.batch, group.target)
            context = group.context[context_order]
            anchor = group.anchor[anchor_order]
            treated = group.treated[treated_order]
            delta_null = context[:n].mean(0) - group.anchor.mean(0)
            first_reference, second_reference = anchor[:16].mean(0), anchor[16:].mean(0)
            halves = (treated[:n1].mean(0) - first_reference, treated[n1:].mean(0) - second_reference)
            nulls = (context[:n1].mean(0) - first_reference, context[n1:n].mean(0) - second_reference)
            actual_half, null_half = half_statistics(*halves), half_statistics(*nulls)
            actual_energy, null_energy = _energy(observed[index]), _energy(delta_null)
            row = {"observed_group_effect_mse": actual_energy, "null_group_effect_mse": null_energy,
                   "excess_group_effect_mse": actual_energy - null_energy}
            for statistic in actual_half:
                row["observed_within_batch_half_" + statistic] = actual_half[statistic]
                row["null_within_batch_half_" + statistic] = null_half[statistic]
                if statistic != "agreement":
                    row["excess_within_batch_half_" + statistic] = actual_half[statistic] - null_half[statistic]
            group_metrics.append(row); group_history[index].append(row)
            null_vectors.append(delta_null); real_halves.append(halves); null_halves.append(nulls)
        targets_this_seed = {}
        for key, indices in target_groups.items():
            record = _mean_rows([group_metrics[index] for index in indices])
            real = np.stack([observed[i] for i in indices]); null = np.stack([null_vectors[i] for i in indices])
            real_pooled, null_pooled = _energy(real.mean(0)), _energy(null.mean(0))
            real_between, null_between = between_batch_crossdot(real), between_batch_crossdot(null)
            record.update(observed_target_pooled_effect_mse=real_pooled, null_target_pooled_effect_mse=null_pooled,
                          excess_target_pooled_effect_mse=real_pooled-null_pooled,
                          observed_between_batch_crossdot=real_between, null_between_batch_crossdot=null_between,
                          excess_between_batch_crossdot=real_between-null_between)
            pooled_real_halves = half_statistics(*(np.mean([real_halves[i][half] for i in indices], axis=0) for half in (0, 1)))
            pooled_null_halves = half_statistics(*(np.mean([null_halves[i][half] for i in indices], axis=0) for half in (0, 1)))
            for statistic in pooled_real_halves:
                record["observed_target_pooled_half_" + statistic] = pooled_real_halves[statistic]
                record["null_target_pooled_half_" + statistic] = pooled_null_halves[statistic]
                if statistic != "agreement":
                    record["excess_target_pooled_half_" + statistic] = pooled_real_halves[statistic] - pooled_null_halves[statistic]
            target_history[key].append(record); targets_this_seed[key] = record
        source_metrics = {source: _mean_rows([row for (s, _), row in targets_this_seed.items() if s == source])
                          for source in sorted({g.source for g in groups})}
        per_seed.append({"seed": seed, "sources": source_metrics, "overall": _mean_rows(list(source_metrics.values()))})
    result = {"schema": SCHEMA, "policy": policy(), "input_counts": input_counts,
            "groups": [{"group": g.index, "source": g.source, "target": g.target, "batch": g.batch,
                        "treated_cells": len(g.treated), "treated_half_sizes": [len(g.treated)//2, len(g.treated)-len(g.treated)//2],
                        "context_cells": 32, "anchor_cells": 32, "anchor_half_sizes": [16, 16],
                        "metrics": _summarize_rows(group_history[i])} for i, g in enumerate(groups)],
            "targets": [{"source": source, "target": target, "batches": len(target_groups[source, target]),
                         "metrics": _summarize_rows(target_history[source, target])} for source, target in cohorts],
            "source_partition_statistics": {source: _summarize_rows([row["sources"][source] for row in per_seed]) for source in per_seed[0]["sources"]},
            "overall": _summarize_rows([row["overall"] for row in per_seed]), "per_seed": per_seed,
            "raw_files_opened": False, "model_outputs_used": False, "fitting_performed": False,
            "independent_biological_repetitions": False, "sham_values_used_in_statistics": False,
            "confidence_intervals_or_pvalues": False, "selection_performed": False}
    compact = lambda metrics: {key: metrics[full] for key, full in AGGREGATE_KEYS.items()}
    result["aggregate"] = compact({key: summary["mean"] for key, summary in result["overall"].items()})
    result["sources"] = [{"source": source,
                          "aggregate": compact({key: summary["mean"] for key, summary in statistics.items()}),
                          "partition_statistics": statistics,
                          "targets": [target for target in result["targets"] if target["source"] == source]}
                         for source, statistics in result["source_partition_statistics"].items()]
    result["seed_summaries"] = [{"seed": row["seed"], "aggregate": compact(row["overall"]),
                                 "sources": [{"source": source, "aggregate": compact(metrics)} for source, metrics in row["sources"].items()]}
                                for row in per_seed]
    return result
