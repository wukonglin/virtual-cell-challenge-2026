"""Expression-free joint target/source role registration and fold-specific GO.

Metadata roles are written before GO source parsing. Existing expression NPZs
are only hashed, never decoded. Four fresh vocabularies use their42 training
targets only, while all56 external target annotations may be transformed.
No downloads, HDF5 access, model fitting, outcome-based target selection, or
modification of previous artifacts is performed by this entry point.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import numpy as np

import build_go_target_features as go
import prepare_public_control_anchor_v2 as v2
from train_public_flow_pilot import authenticated_bytes, authenticated_json, load_go_table, sha256_file, write_new

SCHEMA = "public-joint-target-source-v3"
SEED = 20260920
NAMESPACE = "joint-target-source-v3"
PILOT_RELATIVE = "artifacts/public_flow/control_anchor_v2_20260909"
FILES = {"cache": "cache.npz", "contract": "contract.json", "source_manifest": "sources.json",
         "targets": "targets.json", "prior_suite": "validation/suite.json"}
EXPECTED = {
    "cache_sha256": "13ed1ea60e0dff4d516098e9ef5c4d0520c7847ce2bdfc733082f645a5ee7186",
    "contract_sha256": "a94565b29d14d98588e8dbf58ad25481e1ebc5e20b209db643667d8a5344bbf1",
    "source_manifest_sha256": "b7e7d0a300fe4baeaf8d8963894ef2aa6b6681c1e8cb346f7e31c2515e7a78e7",
    "targets_sha256": "b98438386d1a7dd89d3665e2d538b94ec4f656207a6749ecbc7f28c4148d8eac",
    "prior_suite_sha256": "c6eaae863a037bf0826d3f0384eeb781e4c4be29c1e094e66d5f9a25b917044f",
}
GO_RELATIVE = "dataset/reference/go/2026-08-05/acquisition.json"
GO_EXPECTED = {
    "acquisition_sha256": "ce799783141d4156e1db15e6f7ceedbd044762accae7124de3e3841e62e7ae29",
    "gaf_sha256": "a0afba19dfb1f8fa996bc1bdcd61fd0c9bd4cf0d2bf2509d09ac86993c0e70a2",
    "obo_sha256": "b08d45b268b8c24ccb2513dbbbc7d4df9f6521c099b413f79eb31e06e0fa3bcc",
}
MODULES = ("prepare_public_joint_target_source_v3.py", *v2.DEPENDENCIES)


def policy():
    return {"schema": SCHEMA, "seed": SEED, "namespace": NAMESPACE, "target_folds": 4,
        "targets_total": 56, "held_targets_per_fold": 14, "fit_targets_per_fold": 42,
        "partition": "SHA256 of UTF8 seed|namespace|target, tie exact symbol; consecutive ranked blocks14, roles alphabetically sorted",
        "directions": "two source directions per target fold; fit one source and42targets; evaluate opposite source and14held targets",
        "held_target_exclusion": "held-target treated groups absent from fitting in BOTH sources",
        "group_identity": "exact flattened v2 row-contract source/group order; no new source metadata or expression reads",
        "go_vocabulary": "reparse pinned public GAF/OBO; union direct accepted terms on42fit targets ONLY; transformall56",
        "go_policy": "exact symbols, primary active direct terms, noIEA/ND/NOT, no ancestors/aliases",
        "overlap_audit": "held-versus-training binaryGO Jaccard in FIT vocabulary; ties alphabetic; no outcome weighting/filtering",
        "comparison_scope": "post-hoc public-development experiment on previously exposed v2cells; no untouched test claim",
        "reserved_expression": "RPE1/H1/HepG2/challenge2026/Feng excluded; preserve772v7/prior-target exclusions",
        "expression_decode_during_registration": False, "new_raw_expression": False,
        "automatic_promotion": False, "hyperparameter_selection": False}


def _dump(path, value):
    write_new(Path(path), go._json_bytes(value))


def _codes():
    return {name: sha256_file(Path(__file__).with_name(name)) for name in MODULES}


def _bindings(repo):
    paths = {key: repo / PILOT_RELATIVE / relative for key, relative in FILES.items()}
    actual = {key + "_sha256": sha256_file(path) for key, path in paths.items()}
    if actual != EXPECTED:
        raise ValueError("Only the exact previously exposed v2 inputs may be registered")
    contract = v2.load_contract(paths["contract"], EXPECTED["contract_sha256"], paths["source_manifest"],
                                EXPECTED["source_manifest_sha256"], repo=repo)
    target_bytes = authenticated_bytes(paths["targets"], EXPECTED["targets_sha256"], 1 << 20)
    targets = json.loads(target_bytes)
    if targets != contract["common_targets"] or len(targets) != 56 or targets != sorted(set(targets)):
        raise ValueError("Frozen56-target metadata differs from v2 row registration")
    return {key: str(path.resolve()) for key, path in paths.items()}, contract, targets


def make_folds(targets, roster):
    targets = go._names(targets, "frozen targets")
    if len(targets) != 56:
        raise ValueError("Exactly56 frozen targets required")
    ranked = sorted(targets, key=lambda target: (hashlib.sha256(f"{SEED}|{NAMESPACE}|{target}".encode()).digest(), target))
    folds = []
    for number in range(4):
        fold_id = f"target_fold_{number}"
        held = sorted(ranked[14*number:14*(number+1)])
        fit = sorted(set(targets)-set(held))
        directions = []
        held_all = [g["group_id"] for g in roster if g["target"] in held]
        for fit_source in v2.SOURCE_IDS:
            held_source = next(source for source in v2.SOURCE_IDS if source != fit_source)
            fitted = [g["group_id"] for g in roster if g["source"] == fit_source and g["target"] in fit]
            evaluated = [g["group_id"] for g in roster if g["source"] == held_source and g["target"] in held]
            other = [g["group_id"] for g in roster if g["group_id"] not in set(fitted)|set(held_all)]
            directions.append({"direction_id": f"{fold_id}__{fit_source}__to__{held_source}",
                "fit_source": fit_source, "held_source": held_source,
                "fit_group_ids": fitted, "joint_eval_group_ids": evaluated,
                "held_target_group_ids_all_sources": held_all, "other_excluded_group_ids": other})
        folds.append({"fold_id": fold_id, "fit_targets": fit, "held_targets": held, "directions": directions})
    return folds


def _plan(repo, protocol):
    paths, row_contract, targets = _bindings(repo)
    roster = [{"group_id": index, "source": source["id"], "target": group["target"], "batch": group["batch"]}
              for index, (source, group) in enumerate((source, group) for source in row_contract["sources"] for group in source["groups"])]
    if len(row_contract["excluded_targets"]) != 772 or set(targets)&set(row_contract["excluded_targets"]):
        raise ValueError("Protected target exclusions changed")
    folds = make_folds(targets, roster)
    return {"schema": SCHEMA, "policy": policy(), "inputs": dict(EXPECTED), "input_paths": paths,
        "protocol": {"path": str(protocol.resolve()), "sha256": sha256_file(protocol)},
        "code_sha256": _codes(), "go_source_sha256": dict(GO_EXPECTED),
        "go_acquisition_path": str((repo / GO_RELATIVE).resolve()),
        "excluded_targets": row_contract["excluded_targets"], "targets": targets,
        "group_roster": roster, "folds": folds, "expression_decoded": False,
        "model_fitting_performed": False, "prior_artifacts_modified": False}


def _go_index(acquisition_path):
    """Parse only the exact authenticated source bytes, not a reopened path."""
    receipt = authenticated_json(acquisition_path, GO_EXPECTED["acquisition_sha256"])
    if receipt.get("schema") != "vcc-public-feature-acquisition-v1":
        raise ValueError("Unsupported GO acquisition schema")
    records = receipt.get("files")
    if not isinstance(records, list) or len(records) != 2 or {r.get("path") for r in records} != {"go-basic.obo", "HUMAN-uniprot.gaf.gz"}:
        raise ValueError("Exact two-file GO acquisition required")
    payloads = {}
    for record in records:
        kind = "gaf_sha256" if record["path"].endswith("gz") else "obo_sha256"
        if record.get("sha256") != GO_EXPECTED[kind]:
            raise ValueError("GO source does not match its pinned identity")
        payload = authenticated_bytes(acquisition_path.parent / record["path"], GO_EXPECTED[kind], 64 << 20)
        if len(payload) != record.get("size_bytes"):
            raise ValueError("GO source byte size changed")
        payloads[kind] = payload
    ontology = go.parse_obo(io.StringIO(payloads["obo_sha256"].decode("utf-8")))
    with gzip.GzipFile(fileobj=io.BytesIO(payloads["gaf_sha256"])) as stream:
        with io.TextIOWrapper(stream, encoding="utf-8") as text:
            return go.parse_gaf(text, ontology, include_iea=False)


def coverage_audit(result, fit_targets, held_targets):
    values = result.arrays
    targets = tuple(str(target) for target in values["target_ids"])
    lookup = {target: index for index, target in enumerate(targets)}
    vocabulary = values["features"].astype(bool)
    fitting = sorted(fit_targets)
    train_rows = [lookup[t] for t in fitting]
    if not np.all(vocabulary[train_rows].any(axis=0)):
        raise ValueError("Fold GO vocabulary contains a held-only term")
    roles = {}
    for role, symbols in (("fit", fit_targets), ("held", held_targets)):
        indices = [lookup[t] for t in symbols]
        roles[role] = {"targets": len(symbols), **{key: int(values[key][indices].sum()) for key in ("known_symbol", "has_accepted_annotation", "present")},
                       "missing_in_vocab_targets": [t for t in symbols if not values["present"][lookup[t]]]}
    neighbors = []
    for target in sorted(held_targets):
        row = lookup[target]
        candidates = []
        if values["present"][row]:
            for candidate in fitting:
                other = lookup[candidate]
                if not values["present"][other]:
                    continue
                overlap = int(np.count_nonzero(vocabulary[row] & vocabulary[other]))
                union = int(np.count_nonzero(vocabulary[row] | vocabulary[other]))
                candidates.append((overlap / union, candidate, overlap, union))
        best = sorted(candidates, key=lambda entry: (-entry[0], entry[1]))[0] if candidates else None
        neighbors.append({"target": target, "in_vocabulary_terms": int(values["in_vocab_term_count"][row]),
            "accepted_terms_outside_fit_vocabulary": int(values["accepted_term_count"][row]-values["in_vocab_term_count"][row]),
            "nearest_fit_target": best[1] if best else None, "jaccard": best[0] if best else None,
            "intersection_terms": best[2] if best else None, "union_terms": best[3] if best else None})
    return {"schema": "public-joint-go-coverage-v3", "fit_vocabulary_only": True, "vocabulary_size": vocabulary.shape[1],
            "roles": roles, "held_nearest_fit_overlap": neighbors, "expression_or_outcomes_used": False,
            "selection_or_filtering_performed": False}


def register(repo, output_dir, protocol_path):
    repo, output_dir, protocol_path = Path(repo).resolve(), Path(output_dir).resolve(), Path(protocol_path).resolve()
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh joint-gate registration directory required")
    plan = _plan(repo, protocol_path)
    output_dir.mkdir(mode=0o700)
    _dump(output_dir / "fold_plan.json", plan)  # Freeze roles BEFORE any GO parsing.
    plan_sha = sha256_file(output_dir / "fold_plan.json")
    index = _go_index(repo / GO_RELATIVE)
    (output_dir / "go").mkdir(mode=0o700)
    folds = []
    for fold in plan["folds"]:
        feature_fold_id = f"{SCHEMA}:{fold['fold_id']}:{plan_sha}"
        result = go.build_features(index, target_ids=plan["targets"], train_target_ids=fold["fit_targets"],
                                   fold_id=feature_fold_id, sources=dict(GO_EXPECTED))
        feature_dir = output_dir / "go" / fold["fold_id"]
        receipt = go.write_artifacts(result, feature_dir)
        audit = coverage_audit(result, fold["fit_targets"], fold["held_targets"])
        _dump(feature_dir / "coverage.json", audit)
        folds.append({**fold, "go": {"feature_fold_id": feature_fold_id,
            "npz_path": str(feature_dir / "features.npz"), "npz_sha256": receipt["npz_sha256"],
            "receipt_path": str(feature_dir / "receipt.json"), "receipt_sha256": sha256_file(feature_dir / "receipt.json"),
            "coverage_path": str(feature_dir / "coverage.json"), "coverage_sha256": sha256_file(feature_dir / "coverage.json")}})
    manifest = {**plan, "folds": folds, "fold_plan_path": str(output_dir / "fold_plan.json"), "fold_plan_sha256": plan_sha,
                "fold_go_completed": True, "full56_vocabulary_reused": False}
    _dump(output_dir / "role_manifest.json", manifest)
    completion = {"schema": SCHEMA, "completed": True, "role_manifest_sha256": sha256_file(output_dir / "role_manifest.json"),
                  "fold_plan_sha256": plan_sha, "target_folds": 4, "source_directions": 8,
                  "expression_decoded": False, "response_model_fitted": False, "prior_artifacts_modified": False}
    _dump(output_dir / "complete.json", completion)
    return completion


def load_registration(path, expected_sha256, *, repo=None):
    repo = Path(repo).resolve() if repo is not None else Path(__file__).resolve().parents[1]
    record = authenticated_json(Path(path), expected_sha256)
    if record.get("fold_go_completed") is not True or record.get("full56_vocabulary_reused") is not False:
        raise ValueError("Fold-specific GO construction incomplete or reused full56 vocabulary")
    plan = authenticated_json(Path(record["fold_plan_path"]), record["fold_plan_sha256"])
    expected_plan = _plan(repo, Path(plan["protocol"]["path"]))
    if plan != expected_plan:
        raise ValueError("Frozen target/source roles, inputs, protocol or implementation changed")
    if any(record.get(key) != value for key, value in plan.items() if key != "folds"):
        raise ValueError("Role manifest differs from its frozen metadata plan")
    if set(record) != set(plan)|{"fold_plan_path", "fold_plan_sha256", "fold_go_completed", "full56_vocabulary_reused"}:
        raise ValueError("Unexpected role-manifest fields")
    folds = record.get("folds")
    if not isinstance(folds, list) or len(folds) != 4:
        raise ValueError("Exactly four target folds required")
    root = Path(path).resolve().parent
    for actual, expected in zip(folds, plan["folds"]):
        if set(actual) != set(expected)|{"go"} or any(actual.get(key) != value for key, value in expected.items()):
            raise ValueError("Target/source group roles differ from frozen plan")
        features = actual["go"]
        if features.get("feature_fold_id") != f"{SCHEMA}:{expected['fold_id']}:{record['fold_plan_sha256']}":
            raise ValueError("Fold GO provenance differs")
        for name, filename in (("npz", "features.npz"), ("receipt", "receipt.json"), ("coverage", "coverage.json")):
            target = Path(features[name + "_path"])
            if target != root / "go" / expected["fold_id"] / filename or sha256_file(target) != features[name + "_sha256"]:
                raise ValueError("Fold GO artifact path/hash mismatch")
        receipt = authenticated_json(Path(features["receipt_path"]), features["receipt_sha256"])
        if (receipt.get("fold_id") != features["feature_fold_id"] or receipt.get("sources") != GO_EXPECTED
                or receipt.get("train_target_ids_sha256") != go._sha_json(expected["fit_targets"])
                or receipt.get("target_ids_sha256") != go._sha_json(plan["targets"])
                or receipt.get("vocabulary_fit_scope") != "explicit_training_targets_only"
                or receipt.get("npz_sha256") != features["npz_sha256"]):
            raise ValueError("Fold GO receipt does not bind its42 training targets and56 output rows")
    return record


def load_feature_tables(record):
    tables = {}
    for fold in record["folds"]:
        info = fold["go"]
        tables[fold["fold_id"]] = load_go_table(Path(info["npz_path"]), info["npz_sha256"],
            Path(info["receipt_path"]), info["receipt_sha256"], fold["fit_targets"])
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        raise RuntimeError("Use an approved BioHPC compute host, not an HPC login/head node")
    print(json.dumps(register(args.repo, args.output_dir, args.protocol), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
