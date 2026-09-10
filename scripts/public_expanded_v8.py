"""Sharded execution/replay of frozen v6 science on registered expanded cells.

Only the caller may admit raw data, register roles or promote a model. Each fit
is persisted independently; auditing restores saved heads and never refits.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import socket
import zipfile

import numpy as np

import public_nested_v6 as previous
import public_replicated_cache_v7 as cache_io
import public_training_readiness as safe
from train_public_flow_pilot import write_new

MODEL_SCHEMA = "public-expanded-regularization-model-v8"
SUMMARY_SCHEMA = "public-expanded-regularization-summary-v8"
ARMS = previous.ARMS
SHUFFLED_ARMS = previous.SHUFFLED_ARMS
GRID = previous.GRID
MAX_SHARD_BYTES = 512 << 20
MAX_OUTPUT_BYTES = 32 << 30
MAX_JSON_BYTES = safe.MAX_JSON_BYTES
FALSE_FLAGS = ("count_emitter", "posttraining_performed", "promoted", "submission_performed",
    "H1_treated_read", "RPE1_treated_read", "challenge_treated_used",
    "outer_evaluation_roles_used_for_fitting", "outer_response_selection")
TRUE_FLAGS = ("exact_ntc_identity_verified", "exact_zero_effect_identity_verified",
    "prior_public_development_exposure_acknowledged")
BINDING_KEYS = {"path", "sha256", "compressed_bytes", "expanded_bytes", "array_bytes"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def equal(actual, expected, message):
    require(previous.canonical(actual) == previous.canonical(expected), message)


def policy():
    return {"schema": MODEL_SCHEMA, "science": previous.policy(),
        "cohort_and_index_change": "registered replication-expanded public cohort; reindexed flat group IDs also enter frozen donor ranking",
        "cross_version_comparison": "not a data-volume-only causal comparison; within-v8 arms share the same group IDs",
        "artifact_layout": "per-fit separate weights and predictions NPZ; unprefixed members plus genes",
        "replay": "frozen v6 saved-head replay; no ridge refits",
        "max_shard_compressed_bytes": MAX_SHARD_BYTES,
        "max_shard_expanded_bytes": MAX_SHARD_BYTES,
        "max_output_bytes": MAX_OUTPUT_BYTES,
        "max_json_bytes": MAX_JSON_BYTES,
        "artifact_roster": "exact ordered inner candidates then outer fit for every direction",
        "automatic_promotion": False, "posttraining": False, "count_emitter": False}


def state_keys(arm):
    return {"genes", *previous.COMMON_KEYS,
            *("kernel_" + key for key in previous.kernel.decoder_keys(arm))}


def prediction_keys():
    return {"genes", *previous.PREDICTION_KEYS}


def descriptor(number, outer, phase, candidate_index, regularization):
    require(phase in {"inner", "outer"}, "Invalid shard phase")
    prefix = (f"fold_{number}_inner_{candidate_index}_" if phase == "inner"
              else f"fold_{number}_outer_")
    return {"prefix": prefix, "fold_index": number, "fold_id": outer.fold_id,
            "phase": phase, "candidate_index": candidate_index,
            "regularization": regularization}


def _binding(binding, expected_name):
    require(isinstance(binding, dict) and set(binding) == BINDING_KEYS,
            "Exact shard binding fields required")
    equal(binding["path"], expected_name, "Shard path/phase differs from fixed roster")
    require(Path(expected_name).name == expected_name, "Flat relative shard filename required")
    for key in ("compressed_bytes", "expanded_bytes", "array_bytes"):
        require(type(binding[key]) is int and 0 < binding[key] <= MAX_SHARD_BYTES,
                "Invalid typed shard size budget")
    require(isinstance(binding["sha256"], str) and safe.HASH.fullmatch(binding["sha256"]),
            "Explicit shard SHA-256 required")


def _read_shard(root, binding, keys, expected_name):
    _binding(binding, expected_name)
    # Frozen v7 reader hashes the same nonblocking, no-follow regular handle,
    # validates every ZIP member and NPY allocation header, then decodes bytes.
    arrays, details = cache_io.read_npz(Path(root) / expected_name, binding["sha256"], keys,
                                      return_metadata=True)
    equal({"compressed_bytes": details["cache_bytes"],
           "expanded_bytes": details["expanded_archive_bytes"],
           "array_bytes": details["array_bytes"]},
          {key: binding[key] for key in ("compressed_bytes", "expanded_bytes", "array_bytes")},
          "Shard byte binding changed")
    return arrays


def _save_shard(root, name, arrays, keys, output_bytes):
    require(set(arrays) == keys, "Exact saved shard member roster required")
    require(all(isinstance(value, np.ndarray) and not value.dtype.hasobject
                and value.dtype.fields is None for value in arrays.values()), "Unsafe shard arrays")
    array_bytes = sum(value.nbytes for value in arrays.values())
    # Preflight before compression buffers; 64 KiB conservatively bounds the
    # small registered roster's NPY headers. Exact sizes are checked afterward.
    require(0 < array_bytes + 65536 <= MAX_SHARD_BYTES, "Oversized shard allocation")
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    payload = buffer.getvalue()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        expanded_bytes = sum(info.file_size for info in archive.infolist())
    require(len(payload) <= MAX_SHARD_BYTES and expanded_bytes <= MAX_SHARD_BYTES,
            "Oversized compressed or expanded shard")
    require(output_bytes + len(payload) <= MAX_OUTPUT_BYTES, "Arm output disk budget exceeded")
    binding = {"path": name, "sha256": hashlib.sha256(payload).hexdigest(),
        "compressed_bytes": len(payload), "expanded_bytes": expanded_bytes, "array_bytes": array_bytes}
    write_new(Path(root) / name, payload)
    # Re-read persisted bytes before binding them to a completed summary.
    _read_shard(root, binding, keys, name)
    return binding


def event_record(phase, number, outer, regularization, report):
    return {"phase": phase, "fold_index": number, "held_outer_source": outer.held_source,
            "regularization": regularization, "metrics": report["metrics"],
            "training_diagnostics": report["training_diagnostics"]}


def _folds(arrays, roles, outer_tables, inner_tables):
    folds = previous.v3.joint_folds(tuple(arrays["targets"].astype(str)),
                                    tuple(arrays["sources"].astype(str)), roles)
    require(set(outer_tables) == {fold.target_fold for fold in folds}
            and set(inner_tables) == set(outer_tables), "Exact inner/outer vocabularies required")
    return folds


def _provenance(provenance):
    require(isinstance(provenance, dict) and isinstance(provenance.get("diagnostic_contract_sha256"), str)
            and safe.HASH.fullmatch(provenance["diagnostic_contract_sha256"]),
            "Frozen execution provenance required")
    return json.loads(previous.canonical(provenance))


def run_arm(arrays, roles, outer_tables, nested, inner_tables, arm, output_dir, provenance, progress=None):
    require(arm in ARMS, "Unknown expanded arm")
    provenance = _provenance(provenance)
    folds = _folds(arrays, roles, outer_tables, inner_tables)
    output_dir = Path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh expanded arm output required")
    output_dir.mkdir(mode=0o700)
    reports, artifacts, events = [], [], []
    output_bytes = 0

    def persist(number, outer, phase, index, lam, report, output, state):
        nonlocal output_bytes
        row = descriptor(number, outer, phase, index, lam)
        for kind, values, keys in (("weights", state, state_keys(arm)),
                                   ("predictions", output, prediction_keys())):
            payload = {**values, "genes": arrays["genes"]}
            row[kind] = _save_shard(output_dir, row["prefix"] + kind + ".npz", payload, keys, output_bytes)
            output_bytes += row[kind]["compressed_bytes"]
        artifacts.append(row)
        event = event_record(phase, number, outer, lam, report)
        events.append(event)
        if progress:
            progress(event)

    for number, outer in enumerate(folds):
        candidates = []
        if arm != "control":
            inner = previous.inner_fold(arrays, outer, nested)
            view = previous.inner_view(arrays, inner, nested)
            for index, lam in enumerate(GRID):
                prefix = descriptor(number, outer, "inner", index, lam)["prefix"]
                report, output, state = previous.one_fit(view, inner, inner_tables[outer.target_fold], arm, lam, prefix)
                persist(number, outer, "inner", index, lam, report, output, state)
                candidates.append(report)
                del output, state
            selected = previous.select_regularization(candidates)
            del view
        else:
            selected = 10.0
        prefix = descriptor(number, outer, "outer", None, selected)["prefix"]
        report, output, state = previous.one_fit(arrays, outer, outer_tables[outer.target_fold], arm, selected, prefix)
        persist(number, outer, "outer", None, selected, report, output, state)
        report["inner_selection"] = previous.selection_record(arm, candidates, selected)
        reports.append(report)
        del output, state
    sources, aggregate = previous.sparse._final_metrics(reports)
    steps = b"".join((previous.canonical(event) + "\n").encode() for event in events)
    require(len(steps) <= MAX_JSON_BYTES, "Oversized step record")
    summary = {"schema": SUMMARY_SCHEMA, "status": "completed_unpromoted_diagnostic",
        "stage": "validation", "arm": arm, "policy": policy(), "folds": reports,
        "source_metrics": sources, "aggregate": aggregate, "provenance": provenance,
        "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "artifacts": artifacts, "steps_sha256": hashlib.sha256(steps).hexdigest(),
        "shard_compressed_bytes": output_bytes,
        "shard_expanded_bytes": sum(row[kind]["expanded_bytes"] for row in artifacts for kind in ("weights", "predictions")),
        "npz_shards": 2 * len(artifacts), "fits_completed": len(artifacts),
        "exact_ntc_identity_verified": all(row["exact_ntc_identity_verified"] for row in reports),
        "exact_zero_effect_identity_verified": all(row["exact_zero_effect_identity_verified"] for row in reports),
        "prior_public_development_exposure_acknowledged": True,
        "hyperparameter_selection": arm != "control", "inner_selection_performed": arm != "control",
        "selection_scope": "inner_only_disjoint_reference_rows", **dict.fromkeys(FALSE_FLAGS, False),
        "runtime": {"hostname": socket.gethostname(), "device": "cpu", "numpy_version": np.__version__}}
    # Compact JSON keeps complete group-level evidence below the frozen 16 MiB
    # safe JSON reader bound; no report fields are discarded to achieve this.
    encoded = (previous.canonical(summary) + "\n").encode()
    require(len(encoded) <= MAX_JSON_BYTES, "Oversized expanded summary")
    require(output_bytes + len(steps) + len(encoded) <= MAX_OUTPUT_BYTES, "Arm output disk budget exceeded")
    write_new(output_dir / "steps.jsonl", steps)
    write_new(output_dir / "summary.json", encoded)
    return summary


def audit_arm(output_dir, summary, arrays, roles, outer_tables, nested, inner_tables, provenance):
    """Replay one fit at a time from authenticated shards without ridge refits."""
    root = Path(output_dir)
    require(root.is_dir() and not root.is_symlink(), "Regular arm output directory required")
    provenance = _provenance(provenance)
    summary = json.loads(previous.canonical(summary))
    saved_summary = safe.read_json(root / "summary.json")
    equal(saved_summary, summary, "Passed and persisted summaries differ")
    require(summary.get("schema") == SUMMARY_SCHEMA and summary.get("arm") in ARMS,
            "Invalid expanded summary schema/arm")
    arm = summary["arm"]
    expected = {"status": "completed_unpromoted_diagnostic", "stage": "validation", "policy": policy(),
        "provenance": provenance, "diagnostic_contract_sha256": provenance["diagnostic_contract_sha256"],
        "hyperparameter_selection": arm != "control", "inner_selection_performed": arm != "control",
        "selection_scope": "inner_only_disjoint_reference_rows", **dict.fromkeys(FALSE_FLAGS, False),
        **dict.fromkeys(TRUE_FLAGS, True)}
    for key, value in expected.items():
        equal(summary.get(key), value, "Expanded summary binding changed: " + key)
    runtime = summary.get("runtime")
    require(isinstance(runtime, dict) and set(runtime) == {"hostname", "device", "numpy_version"}
            and all(isinstance(value, str) and value for value in runtime.values())
            and runtime["device"] == "cpu", "Invalid expanded runtime")
    folds = _folds(arrays, roles, outer_tables, inner_tables)
    require(isinstance(summary.get("folds"), list) and len(summary["folds"]) == len(folds),
            "Outer fold report roster changed")
    count = len(folds) * (1 if arm == "control" else 4)
    artifacts = summary.get("artifacts")
    require(isinstance(artifacts, list) and len(artifacts) == count, "Exact fit artifact roster required")
    equal(summary.get("fits_completed"), count, "Fit count changed")
    equal(summary.get("npz_shards"), count * 2, "NPZ shard count changed")
    names = {"summary.json", "steps.jsonl"}
    for number, outer in enumerate(folds):
        prefixes = [descriptor(number, outer, "outer", None, 10.0)["prefix"]]
        if arm != "control":
            prefixes += [descriptor(number, outer, "inner", index, lam)["prefix"] for index, lam in enumerate(GRID)]
        names.update(prefix + kind + ".npz" for prefix in prefixes for kind in ("weights", "predictions"))
    require({path.name for path in root.iterdir()} == names, "Arm directory has missing or unexpected artifacts")
    expected_fields = {"schema", "arm", *expected, "runtime", "folds", "source_metrics", "aggregate",
        "artifacts", "steps_sha256", "shard_compressed_bytes", "shard_expanded_bytes", "npz_shards", "fits_completed"}
    require(set(summary) == expected_fields, "Exact expanded summary fields required")
    # Admit the entire typed shard roster and claimed disk budget before any
    # archive decoding. Replay below independently verifies outer selection.
    planned = []
    for number, (outer, saved) in enumerate(zip(folds, summary["folds"])):
        require(isinstance(saved, dict) and type(saved.get("regularization")) is float
                and saved["regularization"] in GRID, "Invalid outer shard regularization")
        if arm != "control":
            planned.extend(descriptor(number, outer, "inner", index, lam) for index, lam in enumerate(GRID))
        planned.append(descriptor(number, outer, "outer", None, saved["regularization"]))
    claimed_bytes = 0
    for row, intended in zip(artifacts, planned):
        require(isinstance(row, dict) and set(row) == set(intended) | {"weights", "predictions"},
                "Exact fit descriptor fields required")
        equal({key: row[key] for key in intended}, intended, "Fit shard order/role/lambda changed")
        for kind in ("weights", "predictions"):
            _binding(row[kind], intended["prefix"] + kind + ".npz")
            claimed_bytes += row[kind]["compressed_bytes"]
    equal(summary["shard_compressed_bytes"], claimed_bytes, "Compressed shard sum changed")
    require(claimed_bytes <= MAX_OUTPUT_BYTES, "Arm output disk budget exceeded")
    reports, events = [], []
    cursor = 0
    compressed_bytes = expanded_bytes = 0

    def replay(number, outer, phase, index, lam, fit_arrays, fold, table, saved):
        nonlocal cursor, compressed_bytes, expanded_bytes
        row = artifacts[cursor]
        cursor += 1
        expected_row = descriptor(number, outer, phase, index, lam)
        require(isinstance(row, dict) and set(row) == set(expected_row) | {"weights", "predictions"},
                "Exact fit descriptor fields required")
        equal({key: row[key] for key in expected_row}, expected_row, "Fit shard order/role/lambda changed")
        restored = {}
        for kind, keys in (("weights", state_keys(arm)), ("predictions", prediction_keys())):
            values = _read_shard(root, row[kind], keys, row["prefix"] + kind + ".npz")
            require(values["genes"].dtype == arrays["genes"].dtype
                    and np.array_equal(values["genes"], arrays["genes"]), "Shard gene axis changed")
            compressed_bytes += row[kind]["compressed_bytes"]
            expanded_bytes += row[kind]["expanded_bytes"]
            require(compressed_bytes <= MAX_OUTPUT_BYTES, "Arm output disk budget exceeded")
            restored[kind] = {row["prefix"] + key: value for key, value in values.items() if key != "genes"}
        report = previous.replay_entry(fit_arrays, fold, table, arm, lam, row["prefix"], saved,
                                       restored["weights"], restored["predictions"])
        events.append(event_record(phase, number, outer, lam, report))
        return report

    for number, (outer, saved) in enumerate(zip(folds, summary["folds"])):
        require(isinstance(saved, dict) and isinstance(saved.get("inner_selection"), dict),
                "Missing inner selection report")
        original = saved["inner_selection"].get("candidates")
        require(isinstance(original, list) and len(original) == (0 if arm == "control" else len(GRID)),
                "Exact inner candidate report roster required")
        candidates = []
        if arm != "control":
            inner = previous.inner_fold(arrays, outer, nested)
            view = previous.inner_view(arrays, inner, nested)
            for index, lam in enumerate(GRID):
                candidates.append(replay(number, outer, "inner", index, lam, view, inner,
                    inner_tables[outer.target_fold], original[index]))
            selected = previous.select_regularization(candidates)
            del view
        else:
            selected = 10.0
        selection = previous.selection_record(arm, candidates, selected)
        for key in ("performed", "scope", "outer_response_selection", "selected_regularization", "tie_rule"):
            equal(saved["inner_selection"].get(key), selection[key], "Inner selection metadata changed")
        previous.sparse._same_tree(saved["inner_selection"], selection, "inner selection")
        plain = {key: value for key, value in saved.items() if key != "inner_selection"}
        report = replay(number, outer, "outer", None, selected, arrays, outer,
                        outer_tables[outer.target_fold], plain)
        report["inner_selection"] = selection
        reports.append(report)
    require(cursor == count, "Unconsumed fit artifacts")
    equal(summary["shard_compressed_bytes"], compressed_bytes, "Compressed shard sum changed")
    equal(summary["shard_expanded_bytes"], expanded_bytes, "Expanded shard sum changed")
    expected_steps = b"".join((previous.canonical(event) + "\n").encode() for event in events)
    with safe.regular_reader(root / "steps.jsonl") as stream:
        before = cache_io.signature(stream)
        require(before[2] <= MAX_JSON_BYTES, "Oversized step record")
        steps = stream.read(MAX_JSON_BYTES + 1)
        require(cache_io.signature(stream) == before, "Step record changed while reading")
    require(steps == expected_steps, "Step metrics/order differ from replay")
    equal(summary["steps_sha256"], hashlib.sha256(steps).hexdigest(), "Step hash differs")
    require(compressed_bytes + len(steps) + (root / "summary.json").stat().st_size <= MAX_OUTPUT_BYTES,
            "Arm output disk budget exceeded")
    sources, aggregate = previous.sparse._final_metrics(reports)
    previous.sparse._same_tree(summary["source_metrics"], sources, "source metrics")
    previous.sparse._same_tree(summary["aggregate"], aggregate, "aggregate")
    return {"passed": True, "ridge_refits": 0, "saved_head_replay": True, "training_prior_replay": True,
        "kernel_system_verified": True, "training_only_transforms_verified": True,
        "deterministic_donor_replay": True, "original_row_mapping_verified": True,
        "pooled_and_batch_metrics_replayed": True, "training_diagnostics_replayed": True,
        "inner_selection_replayed": True, "disjoint_inner_references_verified": True,
        "inner_candidates_replayed": 0 if arm == "control" else len(GRID) * len(folds),
        "out_of_fold_groups": sum(len(fold.evaluation_indices) for fold in folds),
        "out_of_fold_source_targets": sum(len(fold.held_targets) for fold in folds),
        "sharded_artifacts_verified": True, "exact_artifact_roster_verified": True,
        "shard_headers_verified_before_decode": True, "same_open_checksum_verified": True,
        "output_budget_verified": True, "fits_replayed": count, "npz_shards_verified": count * 2,
        "shard_compressed_bytes": compressed_bytes, "shard_expanded_bytes": expanded_bytes}
