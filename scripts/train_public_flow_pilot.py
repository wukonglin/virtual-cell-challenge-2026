"""Fixed-budget public CRISPRi flow pilot; continuous outputs, never submission.

Run one GO/constant/shuffle arm per fresh directory through the external W&B
wrapper. No raw-source loading, online authentication, dev selection, checkpoint
resume, posttraining, clipping, integer rendering, or automatic promotion.

Cache NPZ schema ``public-flow-pilot-cache-v1``: Unicode scalar schema,
representation, contract_sha256, source_manifest_sha256; unique Unicode genes;
for each train/dev prefix: control[C,G], control_group[C], treated[N,G], group[N],
targets[K], sources[K], context[K,2G]. Expression/context are float32, group IDs
are integers, and strings are Unicode (never object arrays). Controls must be
matched by source/target/exact batch in the authenticated upstream cache builder.
This trainer checks group populations and recomputes control mean/population std;
it cannot reconstruct raw source identities from these compact arrays.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import stat
import time
import zipfile

import numpy as np
import torch

from build_go_target_features import _sha_json
from public_flow_conditioning import FeatureTable, ablate_target_features, align_target_features, fit_target_transform
from public_crispri_flow import ConditionalPublicFlow, FlowConfig, flow_matching_loss, sample_heun

SCHEMA = "public-flow-pilot-cache-v1"
REPRESENTATION = "log1p_cp10k_shared_measured_genes"
SPLIT_FIELDS = ("control", "control_group", "treated", "group", "targets", "sources", "context")
CACHE_KEYS = {"schema", "representation", "contract_sha256", "source_manifest_sha256", "genes"} | {
    f"{split}_{field}" for split in ("train", "dev") for field in SPLIT_FIELDS}
GO_KEYS = {"target_ids", "go_ids", "train_target_ids", "features", "known_symbol", "has_accepted_annotation",
           "present", "accepted_term_count", "in_vocab_term_count", "source_object_count"}


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular non-symlink input")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def authenticated(path, expected):
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("Explicit lowercase SHA-256 required")
    if sha256_file(path) != expected:
        raise ValueError("Input SHA-256 mismatch")


def authenticated_bytes(path, expected, maximum):
    """Parse exactly the bounded bytes authenticated on one regular-file open."""
    if not isinstance(expected, str) or re.fullmatch(r"[a-f0-9]{64}", expected) is None:
        raise ValueError("Explicit lowercase SHA-256 required")
    if path.is_symlink():
        raise ValueError("Expected a regular non-symlink input")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError("Input must be a bounded regular file")
        payload = stream.read(maximum + 1)
    if len(payload) > maximum:
        raise ValueError("Input exceeds its byte-size bound")
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ValueError("Input SHA-256 mismatch")
    return payload


def authenticated_json(path, expected):
    payload = authenticated_bytes(path, expected, 16 << 20)
    def unique_pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result
    def invalid_constant(value):
        raise ValueError("Nonfinite JSON constant")
    result = json.loads(payload, object_pairs_hook=unique_pairs, parse_constant=invalid_constant)
    if not isinstance(result, dict):
        raise ValueError("Expected a JSON object")
    return result


def load_npz(path, expected, keys):
    payload = authenticated_bytes(path, expected, 512 << 20)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != len(keys) or {i.filename for i in members} != {k + ".npy" for k in keys}:
            raise ValueError("NPZ schema/duplicate members mismatch")
        if sum(i.file_size for i in members) > 512 * 1024 * 1024:
            raise ValueError("NPZ expanded size exceeds this bounded pilot")
        for member in members:
            with archive.open(member) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in {(1, 0), (2, 0)}:
                    raise ValueError("Unsupported safe-array NPY format")
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, _, dtype = reader(stream)
                if dtype.hasobject or dtype.fields is not None or any(type(n) is not int or n < 0 for n in shape):
                    raise ValueError("Object/structured arrays or invalid dimensions are forbidden")
                if math.prod(shape) * dtype.itemsize != member.file_size - stream.tell():
                    raise ValueError("NPY declared allocation differs from bounded member payload")
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        values = {key: archive[key] for key in keys}
    if any(value.dtype.hasobject for value in values.values()):
        raise ValueError("Object arrays are forbidden")
    return values


def strings(values, *, unique=False):
    if values.ndim != 1 or not len(values) or values.dtype.kind != "U":
        raise ValueError("Expected a nonempty Unicode string axis")
    result = tuple(str(value) for value in values)
    if any(not value or value.strip() != value for value in result) or (unique and len(set(result)) != len(result)):
        raise ValueError("Malformed or duplicate string axis")
    return result


def scalar(values):
    if values.shape != () or values.dtype.kind != "U":
        raise ValueError("Expected a Unicode scalar declaration")
    return str(values.item())


@dataclass(frozen=True)
class Split:
    control: np.ndarray
    control_group: np.ndarray
    treated: np.ndarray
    group: np.ndarray
    targets: tuple[str, ...]
    sources: tuple[str, ...]
    context: np.ndarray


@dataclass(frozen=True)
class Cache:
    genes: tuple[str, ...]
    train: Split
    dev: Split
    contract_sha256: str
    source_manifest_sha256: str


def validate_split(arrays, prefix, gene_dim):
    data = {field: arrays[f"{prefix}_{field}"] for field in SPLIT_FIELDS}
    targets, sources = strings(data["targets"]), strings(data["sources"])
    groups = len(targets)
    if not 1 <= groups <= 128 or len(sources) != groups:
        raise ValueError("Invalid group/source axis or pilot group bound")
    for name, maximum in (("control", 64), ("treated", 32)):
        matrix = data[name]
        labels = data["control_group" if name == "control" else "group"]
        if matrix.ndim != 2 or matrix.shape[1] != gene_dim or matrix.dtype != np.float32:
            raise ValueError("Cache expression must be float32 cells-by-genes")
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("Cache log1p expression must be finite and nonnegative")
        if labels.shape != (len(matrix),) or labels.dtype.kind not in "iu" or np.any(labels >= groups):
            raise ValueError("Invalid cell-to-group mapping")
        if np.any(labels < 0):
            raise ValueError("Negative group index")
        counts = np.bincount(labels.astype(np.int64), minlength=groups)
        if np.any(counts < 8) or np.any(counts > maximum):
            raise ValueError("Every group needs 8..64 controls and 8..32 treated cells")
    context = data["context"]
    if context.dtype != np.float32 or context.shape != (groups, gene_dim * 2) or not np.isfinite(context).all():
        raise ValueError("Invalid matched-control context features")
    for group in range(groups):
        controls = data["control"][data["control_group"] == group].astype(np.float64)
        expected = np.concatenate((controls.mean(0), controls.std(0)))
        if not np.allclose(context[group], expected, rtol=2e-5, atol=2e-6):
            raise ValueError("Context features differ from this group's matched controls")
    return Split(data["control"], data["control_group"], data["treated"], data["group"],
                 targets, sources, context)


def load_cache(path, expected_sha256, contract_sha256, source_manifest_sha256):
    arrays = load_npz(path, expected_sha256, CACHE_KEYS)
    if scalar(arrays["schema"]) != SCHEMA or scalar(arrays["representation"]) != REPRESENTATION:
        raise ValueError("Wrong cache schema/representation")
    for key, expected in (("contract_sha256", contract_sha256), ("source_manifest_sha256", source_manifest_sha256)):
        if re.fullmatch(r"[a-f0-9]{64}", expected) is None or scalar(arrays[key]) != expected:
            raise ValueError("Cache contract/source-manifest identity mismatch")
    genes = strings(arrays["genes"], unique=True)
    if not 1 <= len(genes) <= 512:
        raise ValueError("This pilot supports at most 512 fully observed shared genes")
    train, dev = (validate_split(arrays, prefix, len(genes)) for prefix in ("train", "dev"))
    if set(train.sources) & set(dev.sources):
        raise ValueError("Development recipient sources must be absent from training")
    if not set(dev.targets) <= set(train.targets):
        raise ValueError("This registered pilot tests same-target recipient-context transfer only")
    return Cache(genes, train, dev, contract_sha256, source_manifest_sha256)


def load_go_table(npz_path, npz_sha256, receipt_path, receipt_sha256, train_targets):
    receipt = authenticated_json(receipt_path, receipt_sha256)
    if receipt.get("schema") != "vcc-go-target-features-v1" or receipt.get("npz_sha256") != npz_sha256:
        raise ValueError("GO receipt/NPZ identity mismatch")
    if receipt.get("npz_size_bytes") != npz_path.stat().st_size:
        raise ValueError("GO receipt/NPZ byte-size mismatch")
    if receipt.get("vocabulary_fit_scope") != "explicit_training_targets_only":
        raise ValueError("GO vocabulary must be training-target-only")
    if receipt.get("policy", {}).get("include_iea") is not False or receipt.get("policy", {}).get("ancestor_policy") != "direct_only":
        raise ValueError("Registered GO pilot excludes IEA and ancestor propagation")
    arrays = load_npz(npz_path, npz_sha256, GO_KEYS)
    targets, vocabulary, train = (strings(arrays[key], unique=True) for key in
                                  ("target_ids", "go_ids", "train_target_ids"))
    if set(train) != set(train_targets) or receipt.get("train_target_ids_sha256") != _sha_json(train):
        raise ValueError("GO vocabulary training IDs differ from this cache's training targets")
    if not set(train) <= set(targets) or receipt.get("vocabulary_sha256") != _sha_json(vocabulary):
        raise ValueError("GO target/vocabulary axis mismatch")
    if any(re.fullmatch(r"GO:[0-9]{7}", term) is None for term in vocabulary):
        raise ValueError("Malformed GO vocabulary")
    features = arrays["features"]
    if features.dtype != np.uint8 or features.shape != (len(targets), len(vocabulary)) or np.any(features > 1):
        raise ValueError("GO features must be binary uint8 on the declared axes")
    for key in ("present", "known_symbol", "has_accepted_annotation"):
        if arrays[key].dtype != np.bool_ or arrays[key].shape != (len(targets),):
            raise ValueError("Explicit GO coverage masks required")
    if not np.array_equal(arrays["present"], features.any(axis=1)):
        raise ValueError("GO present mask disagrees with in-vocabulary features")
    if np.any(arrays["present"] & ~arrays["has_accepted_annotation"]) or np.any(arrays["has_accepted_annotation"] & ~arrays["known_symbol"]):
        raise ValueError("Contradictory GO coverage masks")
    for key in ("accepted_term_count", "in_vocab_term_count", "source_object_count"):
        if arrays[key].shape != (len(targets),) or arrays[key].dtype.kind not in "iu" or np.any(arrays[key] < 0):
            raise ValueError("Invalid GO annotation/coverage counts")
    if not np.array_equal(arrays["in_vocab_term_count"], features.sum(1)) or np.any(arrays["accepted_term_count"] < arrays["in_vocab_term_count"]):
        raise ValueError("GO term counts disagree with feature coverage")
    if not np.array_equal(arrays["source_object_count"] > 0, arrays["known_symbol"]) or not np.array_equal(arrays["accepted_term_count"] > 0, arrays["has_accepted_annotation"]):
        raise ValueError("GO annotation counts disagree with coverage masks")
    keep = arrays["present"]
    table = FeatureTable("go", tuple(t for t, present in zip(targets, keep) if present), vocabulary,
                         features[keep], receipt["feature_table_source_id"], receipt["feature_table_source_sha256"])
    if table.fingerprint != receipt.get("feature_table_fingerprint"):
        raise ValueError("GO FeatureTable fingerprint mismatch")
    return table


class BalancedSampler:
    """Uniform source -> target -> matching group -> independent cells."""
    def __init__(self, split):
        self.split = split
        self.sources = tuple(sorted(set(split.sources)))
        self.targets = {s: tuple(sorted({t for t, source in zip(split.targets, split.sources) if source == s})) for s in self.sources}
        self.groups = {(s, t): np.array([i for i, (target, source) in enumerate(zip(split.targets, split.sources))
                                         if (source, target) == (s, t)]) for s in self.sources for t in self.targets[s]}
        self.controls = [np.flatnonzero(split.control_group == i) for i in range(len(split.targets))]
        self.treated = [np.flatnonzero(split.group == i) for i in range(len(split.targets))]

    def draw(self, rng, size):
        groups = []
        for _ in range(size):
            source = self.sources[int(rng.integers(len(self.sources)))]
            targets = self.targets[source]
            target = targets[int(rng.integers(len(targets)))]
            groups.append(int(rng.choice(self.groups[source, target])))
        controls = np.array([rng.choice(self.controls[g]) for g in groups])
        treated = np.array([rng.choice(self.treated[g]) for g in groups])
        return np.asarray(groups), controls, treated


def rbf_mmd2(x, y):
    """Registered biased MMD: exp(-mean_squared_gene_distance/2), diagonal included."""
    if min(len(x), len(y)) < 2:
        raise ValueError("MMD needs >=2 cells per population")
    def kernel(a, b):
        a, b = a.astype(np.float64), b.astype(np.float64)
        distance = np.maximum((a * a).sum(1)[:, None] + (b * b).sum(1)[None, :] - 2 * a @ b.T, 0)
        return np.exp(-distance / (2 * a.shape[1]))
    xx, yy, xy = kernel(x, x), kernel(y, y), kernel(x, y)
    return float(xx.mean() + yy.mean() - 2 * xy.mean())


def diagnostics(predicted, control, treated):
    predicted, control, treated = (a.astype(np.float64) for a in (predicted, control, treated))
    if any(not np.isfinite(a).all() for a in (predicted, control, treated)):
        raise ValueError("Nonfinite rollout diagnostics")
    variance = float(treated.var(0).mean())
    singular = np.linalg.svd(predicted - predicted.mean(0), compute_uv=False) ** 2
    total = singular.sum()
    proportions = singular[singular > 0] / total if total else np.array([])
    return {"model_centroid_mse": float(np.square(predicted.mean(0) - treated.mean(0)).mean()),
            "control_centroid_mse": float(np.square(control.mean(0) - treated.mean(0)).mean()),
            "model_mmd2": rbf_mmd2(predicted, treated),
            "control_mmd2": rbf_mmd2(control, treated),
            "variance_ratio": float(predicted.var(0).mean() / variance) if variance > 1e-12 else None,
            "covariance_effective_rank": float(np.exp(-(proportions * np.log(proportions)).sum())) if total else 0.0,
            "zero_fraction": float(np.mean(predicted == 0)),
            "negative_fraction": float(np.mean(predicted < 0)),
            "duplicate_fraction": 1 - len(np.unique(predicted, axis=0)) / len(predicted)}


@dataclass(frozen=True)
class PilotConfig:
    steps: int = 200
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 1.0
    hidden_size: int = 128
    layers: int = 2
    seed: int = 20260909
    integration_steps: int = 16
    feature_mode: str = "true"
    shuffle_seed: int = 20260910

    def __post_init__(self):
        for key, cap in (("steps", 1000), ("batch_size", 256), ("hidden_size", 256), ("layers", 4), ("integration_steps", 64)):
            value = getattr(self, key)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{key} exceeds the bounded pilot contract")
        if type(self.learning_rate) not in (int, float) or not np.isfinite(self.learning_rate) or not 0 < self.learning_rate <= .01:
            raise ValueError("Invalid learning rate")
        if type(self.weight_decay) not in (int, float) or not np.isfinite(self.weight_decay) or not 0 <= self.weight_decay <= .1:
            raise ValueError("Invalid weight decay")
        if type(self.gradient_clip_norm) not in (int, float) or not np.isfinite(self.gradient_clip_norm) or not 0 < self.gradient_clip_norm <= 10:
            raise ValueError("Invalid gradient clipping bound")
        if self.feature_mode not in {"true", "constant", "shuffled"}:
            raise ValueError("Unknown feature arm")
        if any(type(s) is not int or not 0 <= s < 2 ** 31 for s in (self.seed, self.shuffle_seed)):
            raise ValueError("Invalid seed")


def validate_registration(cache, config, contract_path, source_path):
    """Authenticate metadata contracts, never open a raw H5AD/source receipt."""
    contract = authenticated_json(contract_path, cache.contract_sha256)
    manifest = authenticated_json(source_path, cache.source_manifest_sha256)
    if contract.get("schema") != "public-flow-row-contract-v1" or manifest.get("schema") != "public-flow-source-manifest-v1":
        raise ValueError("Unsupported row contract/source manifest")
    if contract.get("representation") != REPRESENTATION or contract.get("source_manifest_sha256") != cache.source_manifest_sha256:
        raise ValueError("Contract representation/source-manifest identity differs")
    profile = contract.get("source_profile", "rpe1_dev")
    if profile not in {"rpe1_dev", "h1_dev"} or manifest.get("source_profile", "rpe1_dev") != profile:
        raise ValueError("Unknown or inconsistent registered source profile")
    if contract.get("recipient_treated_allowed") is not False or contract.get("challenge_2026_treated_allowed") is not False:
        raise ValueError("Recipient/challenge treated cells cannot be admitted to fitting")
    output = strings(np.asarray(contract.get("output_genes")), unique=True)
    shared = strings(np.asarray(contract.get("shared_genes")), unique=True)
    if output != cache.genes or not set(output) <= set(shared):
        raise ValueError("Cache modeled genes differ from the frozen measured-gene axis")
    policy = contract.get("policy", {})
    if not isinstance(policy, dict):
        raise ValueError("Preparation policy must be an object")
    required_policy = {"max_targets_per_source": 16, "max_batches_per_target": 2, "min_cells": 8,
                       "max_controls_per_group": 64, "max_treated_per_group": 32,
                       "matching": "exact source and batch; labeled non-targeting only",
                       "genes": "unique shared symbols; fixed SHA256-ranked subset; no effect selection",
                       "normalization": "CP10000 over full shared measured axis, then log1p, then output subset",
                       "go": "direct primary terms; exact symbols; no IEA/ND/NOT/obsolete/alternate; fit vocabulary train only",
                       "optimizer": "fresh per-arm AdamW; no prior weights; fixed steps; no dev checkpoint selection"}
    if any(type(policy.get(key)) is not type(value) or policy[key] != value for key, value in required_policy.items()):
        raise ValueError("Unsupported or changed frozen preparation policy")
    defaults, actual, training = asdict(PilotConfig()), asdict(config), policy.get("training", {})
    if not isinstance(training, dict):
        raise ValueError("Training policy must be an object")
    keys = ("steps", "batch_size", "hidden_size", "layers", "learning_rate", "weight_decay", "gradient_clip_norm", "seed", "shuffle_seed")
    if any(type(training.get(key)) is not type(defaults[key]) or training[key] != defaults[key] or actual[key] != training[key] for key in keys):
        raise ValueError("Actual pretraining hyperparameters differ from the fixed registered pilot")
    if contract.get("seed") != defaults["seed"] or training.get("arms") != ["true", "constant", "shuffled"] or training.get("sampler") != "uniform source then target then matched batch; unpaired cells":
        raise ValueError("Frozen arm/seed/sampling policy mismatch")
    evaluation = {"scope": "development only; no score claim", "when": "fixed final step only",
                  "heun_steps": 16, "clipping": False,
                  "centroid_mse": "mean over modeled genes of squared population-mean difference",
                  "variance_ratio": "mean predicted population gene variance / mean treated population gene variance",
                  "mmd2": "biased RBF MMD; kernel exp(-mean gene squared distance / 2); bandwidth fixed at 1 on log1p CP10k",
                  "scale": REPRESENTATION}
    if policy.get("evaluation") != evaluation or config.integration_steps != evaluation["heun_steps"]:
        raise ValueError("Frozen final-only evaluation policy mismatch")
    sources, registered = manifest.get("sources"), contract.get("sources")
    if not isinstance(sources, list) or not isinstance(registered, list) or len(sources) != 3 or len(registered) != 3:
        raise ValueError("This pilot requires exactly three explicit sources")
    if any(not isinstance(s, dict) for s in sources + registered):
        raise ValueError("Invalid source specification")
    ids = strings(np.asarray([s.get("id") for s in sources]), unique=True)
    if tuple(s.get("id") for s in registered) != ids:
        raise ValueError("Source roster/order differs from its manifest")
    if sorted(s.get("role", "") for s in sources) != ["dev", "train", "train"]:
        raise ValueError("Expected two fitting sources and one development recipient")
    for key in ("path", "sha256"):
        strings(np.asarray([s.get(key) for s in sources]), unique=True)
    denied = set(strings(np.asarray(contract.get("excluded_targets")), unique=True))
    expected = {role: {"targets": [], "sources": [], "controls": [], "treated": []} for role in ("train", "dev")}
    for source, spec in zip(sources, registered):
        if any(spec.get(key) != value for key, value in source.items()):
            raise ValueError("Contract source identity/role differs from the authenticated source manifest")
        for key in ("sha256", "receipt_sha256", "metadata_sha256"):
            if not isinstance(spec.get(key), str) or re.fullmatch(r"[a-f0-9]{64}", spec[key]) is None:
                raise ValueError("Missing registered source/receipt/metadata digest")
        cells, groups = spec.get("cells_total"), spec.get("groups")
        if type(cells) is not int or cells < 1 or not isinstance(groups, list) or not groups:
            raise ValueError("Invalid source row/group metadata")
        seen, control_rows, treated_rows, per_target = set(), set(), set(), {}
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError("Malformed registered group")
            target, batch = strings(np.asarray([group.get("target"), group.get("batch")]))
            if target in denied or target == spec.get("control_label") or (target, batch) in seen:
                raise ValueError("Excluded target or duplicate matching group")
            seen.add((target, batch))
            per_target[target] = per_target.get(target, 0) + 1
            bucket = expected[spec["role"]]
            bucket["targets"].append(target)
            bucket["sources"].append(spec["id"])
            for field, cap, label in (("control_rows", 64, "controls"), ("treated_rows", 32, "treated")):
                rows = group.get(field)
                if not isinstance(rows, list) or not 8 <= len(rows) <= cap or any(type(row) is not int or not 0 <= row < cells for row in rows):
                    raise ValueError("Invalid registered row list/bounds")
                if rows != sorted(set(rows)):
                    raise ValueError("Registered row indices must be sorted and unique")
                if label == "treated" and treated_rows.intersection(rows):
                    raise ValueError("Repeated treated rows across matching groups")
                (control_rows if label == "controls" else treated_rows).update(rows)
                bucket[label].append(len(rows))
        if control_rows & treated_rows or len(per_target) > 16 or max(per_target.values()) > 2:
            raise ValueError("Source matching/target sampling bounds violated")
    for role, bucket in expected.items():
        split = getattr(cache, role)
        if split.targets != tuple(bucket["targets"]) or split.sources != tuple(bucket["sources"]):
            raise ValueError("Cache group/source/target axes differ from the authorized row contract")
        for labels, lengths in ((split.control_group, bucket["controls"]), (split.group, bucket["treated"])):
            if not np.array_equal(labels, np.repeat(np.arange(len(lengths)), lengths)):
                raise ValueError("Cache row-group lengths/order differ from the authorized contract")
    return {"source_profile": profile, "source_roles": {s["id"]: s["role"] for s in sources},
            "registration_authenticated": True, "raw_source_files_opened_by_trainer": False}


def target_conditions(cache, table, config):
    targets = tuple(sorted(set(cache.train.targets)))
    aligned = align_target_features(targets, [table])
    if config.feature_mode != "true":
        aligned = ablate_target_features(aligned, mode="shuffle" if config.feature_mode == "shuffled" else "constant",
                                         seed=config.shuffle_seed, train_ids=targets, fold_id=cache.contract_sha256)
    fitted = fit_target_transform(aligned, train_ids=targets, fold_id=cache.contract_sha256)
    features = np.column_stack((fitted.transform(aligned), aligned.present, aligned.unknown_target)).astype(np.float32)
    if not np.isfinite(features).all():
        raise ValueError("Nonfinite target conditions")
    lookup = {target: index for index, target in enumerate(targets)}
    by_split = {name: features[[lookup[t] for t in getattr(cache, name).targets]] for name in ("train", "dev")}
    return by_split, fitted, int(aligned.present.sum())


def evaluate(model, split, target_features, *, device, seed, steps):
    rng, groups = np.random.default_rng(seed), []
    for group, (target, source) in enumerate(zip(split.targets, split.sources)):
        control_rows = np.flatnonzero(split.control_group == group)
        treated_rows = np.flatnonzero(split.group == group)
        n = min(32, len(control_rows), len(treated_rows))
        control = split.control[rng.choice(control_rows, n, replace=False)]
        treated = split.treated[rng.choice(treated_rows, n, replace=False)]
        x = torch.as_tensor(control, device=device)
        target_tensor = torch.as_tensor(np.repeat(target_features[group:group + 1], n, axis=0), device=device)
        context = torch.as_tensor(np.repeat(split.context[group:group + 1], n, axis=0), device=device)
        predicted = sample_heun(model, x, target_tensor, context, torch.ones_like(x, dtype=torch.bool), steps=steps).cpu().numpy()
        groups.append({"group": group, "source": source, "target": target, "cells": n,
                       **diagnostics(predicted, control, treated)})
    # Same hierarchy as sampling; absent/undefined metrics are reported, not zero-filled.
    aggregate = {}
    keys = set(groups[0]) - {"group", "source", "target", "cells"}
    for key in sorted(keys):
        source_values = []
        for source in sorted(set(split.sources)):
            target_values = []
            for target in sorted({row["target"] for row in groups if row["source"] == source}):
                values = [row[key] for row in groups if (row["source"], row["target"]) == (source, target) and row[key] is not None]
                if values:
                    target_values.append(float(np.mean(values)))
            if target_values:
                source_values.append(float(np.mean(target_values)))
        aggregate[key] = float(np.mean(source_values)) if source_values else None
    return groups, aggregate


def write_new(path, payload):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def save_weights(path, model, transform):
    arrays = {"model." + key: value.detach().cpu().numpy() for key, value in model.state_dict().items()}
    arrays.update(target_mean=transform.mean, target_scale=transform.scale)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for name, array in sorted(arrays.items()):
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            entry = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, member.getvalue())
    write_new(path, buffer.getvalue())
    return sha256_file(path)


def train_pilot(cache, table, config, output_dir, *, device="cpu", provenance=None):
    """Train on cache.train only; cache.dev is used once after fixed training."""
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh output directory required; resume/posttraining is not implemented")
    output_dir.mkdir(mode=0o700)
    condition, transform, coverage = target_conditions(cache, table, config)
    model_config = FlowConfig(len(cache.genes), condition["train"].shape[1], len(cache.genes) * 2,
                              config.hidden_size, config.layers)
    torch_device = torch.device(device)
    devices = [] if torch_device.type == "cpu" else [torch_device.index or 0]
    sampler, rng = BalancedSampler(cache.train), np.random.default_rng(config.seed)
    started = time.monotonic()
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed)
        model = ConditionalPublicFlow(model_config).to(torch_device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        with (output_dir / "steps.jsonl").open("x") as journal:
            for step in range(1, config.steps + 1):
                group, ci, ti = sampler.draw(rng, config.batch_size)
                control = torch.as_tensor(cache.train.control[ci], device=torch_device)
                treated = torch.as_tensor(cache.train.treated[ti], device=torch_device)
                target = torch.as_tensor(condition["train"][group], device=torch_device)
                context = torch.as_tensor(cache.train.context[group], device=torch_device)
                clock = torch.as_tensor(rng.random((config.batch_size, 1)).astype(np.float32), device=torch_device)
                optimizer.zero_grad(set_to_none=True)
                loss = flow_matching_loss(model, control, treated, target, context, torch.ones_like(control, dtype=torch.bool), time=clock)
                loss.backward()
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_norm, error_if_nonfinite=True)
                optimizer.step()
                record = {"step": step, "loss": float(loss.detach()), "cfm": float(loss.detach()),
                          "gradient_norm": float(gradient), "learning_rate": config.learning_rate,
                          "elapsed_seconds": time.monotonic() - started}
                encoded = json.dumps(record, sort_keys=True, allow_nan=False)
                journal.write(encoded + "\n")
                journal.flush()
                os.fsync(journal.fileno())
                print(encoded, flush=True)
        weights_sha = save_weights(output_dir / "weights.npz", model, transform)
        groups, aggregate = evaluate(model, cache.dev, condition["dev"], device=torch_device,
                                     seed=config.seed + 1, steps=config.integration_steps)
    rollout = {"kind_metrics": {"context": aggregate}}
    summary = {"schema": "vcc-public-flow-pilot-summary-v1", "status": "completed_unpromoted_pilot",
               "stage": "pretrain", "last_step": config.steps, "configuration": asdict(config),
               "model_config": asdict(model_config), "weights_sha256": weights_sha,
               "contract_sha256": cache.contract_sha256, "source_manifest_sha256": cache.source_manifest_sha256,
               "target_transform": transform.provenance(), "go_targets_present": coverage,
               "provenance": provenance or {}, "representation": REPRESENTATION, "output_gene_count": len(cache.genes),
               "runtime": {"torch_version": torch.__version__, "numpy_version": np.__version__,
                           "device": str(torch_device), "hostname": socket.gethostname(),
                           "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                           "gpu_name": torch.cuda.get_device_name(torch_device) if torch_device.type == "cuda" else None},
               "dev_groups": groups, "deployment_rollout_diagnostics": rollout,
               "mmd_bandwidth_squared": 1.0, "mmd_bandwidth_scope": "fixed_registered_not_fitted",
               "mmd_kernel": "exp(-mean_squared_gene_distance/2)", "mmd_estimator": "biased_including_diagonal",
               "context_transform": "raw_matched_control_mean_population_std_verified",
               "sampling": "uniform_source_then_target_then_matching_group_then_independent_cells",
               "dev_used_for_fitting_or_selection": False, "checkpoint_selection": "fixed_final_step_only",
               "count_emitter": False, "full_axis_submission_ready": False, "promotion_decision": "not_performed",
               "posttraining_or_resume_supported": False, "feature_ablation_preserves_coverage_masks": True}
    write_new(output_dir / "summary.json", (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    print(json.dumps({"step": config.steps, "deployment_rollout_diagnostics": rollout}, allow_nan=False), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache", "target-features", "feature-receipt", "output-dir", "contract", "source-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("cache-sha256", "target-features-sha256", "feature-receipt-sha256", "contract-sha256", "source-manifest-sha256"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--stage", choices=("pretrain",), default="pretrain")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    for name, default in asdict(PilotConfig()).items():
        parser.add_argument("--" + name.replace("_", "-"), type=type(default), default=default)
    args = parser.parse_args(argv)
    config = PilotConfig(**{name: getattr(args, name) for name in asdict(PilotConfig())})
    host = socket.gethostname().split(".")[0]
    if not os.environ.get("SLURM_JOB_ID") and host not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Use a Slurm allocation or an approved BioHPC compute host, not a head node")
    if args.device == "cuda":
        if not os.environ.get("SLURM_JOB_ID") or not torch.cuda.is_available() or "H100" not in torch.cuda.get_device_name(0):
            raise RuntimeError("CUDA pilot requires an allocated H100; no CPU fallback")
    cache = load_cache(args.cache, args.cache_sha256, args.contract_sha256, args.source_manifest_sha256)
    registration = validate_registration(cache, config, args.contract, args.source_manifest)
    table = load_go_table(args.target_features, args.target_features_sha256, args.feature_receipt,
                          args.feature_receipt_sha256, cache.train.targets)
    train_pilot(cache, table, config, args.output_dir, device=args.device, provenance={
        "cache_sha256": args.cache_sha256, "target_features_sha256": args.target_features_sha256,
        "feature_receipt_sha256": args.feature_receipt_sha256, "entrypoint_sha256": sha256_file(Path(__file__)),
        "flow_module_sha256": sha256_file(Path(__file__).with_name("public_crispri_flow.py")),
        "conditioning_module_sha256": sha256_file(Path(__file__).with_name("public_flow_conditioning.py")),
        "go_builder_sha256": sha256_file(Path(__file__).with_name("build_go_target_features.py")),
        **registration,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
