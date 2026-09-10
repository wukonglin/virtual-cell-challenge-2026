"""Expression-free nested targets and globally disjoint reference registration.

The fixed v5 parent binds original v2 rows and v3 outer roles. New inner roles
are sealed before parsing pinned public GO. Only the four NEW GO tables may
be decoded by load_feature_tables; no expression NPZ, raw-cell source,
credential, or network service is opened. Legacy parent validation hashes
opaque cache bytes but never decodes them; legacy helpers remain unchanged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import build_go_target_features as go
import prepare_public_joint_target_source_v3 as preparation
import public_training_readiness as evidence
import run_public_target_context_v5 as parent
from train_public_flow_pilot import load_go_table

SCHEMA = "public-nested-target-reference-registration-v6"
PRIOR_SHA = "26e03fa1563f783e4c2fedfc74b38ae100f1e8943665da90474a97c0ef0a757f"
SEED = 20260925
MODULES = tuple(dict.fromkeys(("prepare_public_nested_v6.py", *parent.MODULES)))
SOURCES = {"replogle_k562", "nadig_jurkat"}


def policy():
    return {"schema": SCHEMA, "seed": SEED,
        "hash_framing": "UTF8 compact JSON array; ensure_ascii=False; SHA256 digest ascending, identity tie-break",
        "inner_target_hash": "[seed,nested-target-v6,outer_fold_id,target]",
        "inner_target_roles": "first14 ranked outer-fit targets tune; remaining28 fit; serialized roles alphabetic",
        "anchor_hash": "[seed,nested-anchor-v6,source,batch,original_row]",
        "anchor_roles": "first16 ranked original32 anchor rows fit_reference; remaining16 tuning_anchor; serialize numeric ascending",
        "anchor_scope": "one source/batch global split reused across ALL targets, arms, candidates and folds",
        "inner_directions": "inner28 fit and inner14 tune within original fitting source only; opposite source untouched by selection",
        "reference_use": "inner fitting response/prior uses fit_reference16 only; inner tuning prediction uses disjoint tuning_anchor16",
        "outer_evaluation": "original four42/14 target folds and eight source directions unchanged",
        "go_vocabulary": "four new vocabularies fitted on respective inner28 targets only; transform outer42, never outer-held14",
        "go_policy": "pinned public direct accepted terms; exact symbols; no IEA/ND/NOT, ancestors or alias guessing",
        "lambda_candidates": [0.1, 1.0, 10.0],
        "exposure": "previously exposed public development cells; not an untouched test",
        "exclusions": "preserve772 exclusions; RPE1/H1/HepG2/Feng/challenge2026 expression excluded",
        "expression_decode_during_registration": False, "new_raw_expression": False,
        "response_model_fitting": False, "automatic_promotion": False, "submission": False}


def same_json(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(right, sort_keys=True, allow_nan=False)


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def _canonical(path):
    path = Path(path).absolute()
    require(not path.is_symlink(), "Nonsymlink artifact leaf required")
    # /fs and /NFS4 are supported directory aliases; artifact leaves are not.
    return path.resolve()


def _dump(path, value):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(go._json_bytes(value)); stream.flush(); os.fsync(stream.fileno())


def _rank(namespace, *identity):
    framed = json.dumps([SEED, namespace, *identity], ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(framed.encode("utf-8")).digest()


def _names(values, count, label):
    require(isinstance(values, list) and len(values) == count
        and all(isinstance(x, str) and x and x == x.strip() for x in values)
        and values == sorted(set(values)), "Invalid sorted unique " + label)
    return values


def make_inner_folds(outer_roles):
    targets = _names(outer_roles["targets"], 56, "outer targets")
    folds = outer_roles["folds"]
    require(isinstance(folds, list) and len(folds) == 4, "Four outer folds required")
    roster = outer_roles["group_roster"]
    require(isinstance(roster, list) and all(type(g.get("group_id")) is int and g["group_id"] == i
        and g.get("source") in SOURCES and g.get("target") in targets
        and isinstance(g.get("batch"), str) and g["batch"] for i, g in enumerate(roster)), "Invalid outer group roster")
    result, outer_held = {}, []
    for outer in folds:
        fold_id = outer["fold_id"]
        require(fold_id in {f"target_fold_{i}" for i in range(4)} and fold_id not in result, "Invalid outer fold ID")
        available = _names(outer["fit_targets"], 42, "outer fitting targets")
        held = _names(outer["held_targets"], 14, "outer held targets")
        require(not set(available) & set(held) and sorted(available + held) == targets, "Outer target role overlap")
        outer_held += held
        ranked = sorted(available, key=lambda target: (_rank("nested-target-v6", fold_id, target), target))
        tuning, fit = sorted(ranked[:14]), sorted(ranked[14:])
        directions = []
        require(len(outer["directions"]) == 2 and {d["fit_source"] for d in outer["directions"]} == SOURCES,
            "Two registered source directions required")
        for direction in outer["directions"]:
            source = direction["fit_source"]
            held_source = next(s for s in SOURCES if s != source)
            require(direction["held_source"] == held_source, "Outer source roles overlap")
            fitted = [g["group_id"] for g in roster if g["source"] == source and g["target"] in available]
            evaluated = [g["group_id"] for g in roster if g["source"] == held_source and g["target"] in held]
            require(same_json(direction["fit_group_ids"], fitted)
                and same_json(direction["joint_eval_group_ids"], evaluated), "Outer group role mismatch")
            directions.append({"direction_id": direction["direction_id"], "fit_source": source,
                "held_source": held_source,
                "fit_group_ids": [g["group_id"] for g in roster if g["source"] == source and g["target"] in fit],
                "tuning_group_ids": [g["group_id"] for g in roster if g["source"] == source and g["target"] in tuning],
                "outer_fit_group_ids": fitted, "outer_joint_eval_group_ids": evaluated})
        result[fold_id] = {"outer_fold_id": fold_id, "fit_targets": fit, "tuning_targets": tuning,
            "outer_fit_targets": available, "outer_held_targets": held, "directions": directions}
    require(sorted(outer_held) == targets, "Each target must be outer-held exactly once")
    return {key: result[key] for key in sorted(result)}


def make_anchor_pools(row_contract, outer_roles):
    """Use original SOURCE row IDs, never flattened cache offsets or target IDs."""
    source_records = row_contract["sources"]
    require(len(source_records) == 2 and {s["id"] for s in source_records} == SOURCES,
        "Exactly both original row sources required")
    pools, roster, seen_identity = {}, [], {}
    for source in source_records:
        source_id = source["id"]
        cells = source["cells_total"]
        require(type(cells) is int and cells > 0, "Invalid original source cell count")
        for group in source["groups"]:
            batch = group["batch"]
            require(isinstance(batch, str) and bool(batch), "Invalid original batch")
            roster.append({"group_id": len(roster), "source": source_id, "target": group["target"], "batch": batch})
            value = {}
            for name, count in (("control_rows", 32), ("context_control_rows", 32), ("sham_control_rows", 8)):
                rows = group[name]
                require(isinstance(rows, list) and len(rows) == count
                    and all(type(row) is int and 0 <= row < cells for row in rows)
                    and rows == sorted(set(rows)), "Invalid original control row list")
                value[name] = rows
            require(sum(len(rows) for rows in value.values()) == len(set().union(*map(set, value.values()))),
                "Original context/anchor/sham roles overlap")
            key = (source_id, batch)
            require(key not in pools or same_json(pools[key], value), "Control pool differs within source/batch")
            pools[key] = value
            for field, rows in value.items():
                for row in rows:
                    identity = (source_id, row)
                    require(identity not in seen_identity or seen_identity[identity] == (batch, field),
                        "Original control identity crosses batch/pool roles")
                    seen_identity[identity] = (batch, field)
    require(same_json(roster, outer_roles["group_roster"]), "Original row contract and outer group roster differ")
    result = []
    for (source, batch), value in sorted(pools.items()):
        ranked = sorted(value["control_rows"], key=lambda row: (_rank("nested-anchor-v6", source, batch, row), row))
        result.append({"source": source, "batch": batch, "fit_reference_rows": sorted(ranked[:16]),
            "tuning_anchor_rows": sorted(ranked[16:]), "context_rows": value["context_control_rows"],
            "sham_rows": value["sham_control_rows"]})
    return result


def _codes():
    return {name: evidence.digest(Path(__file__).with_name(name)) for name in MODULES}


def _plan(prior_path, protocol):
    prior_path, protocol = _canonical(prior_path), _canonical(protocol)
    sealed = evidence.read_json(prior_path, PRIOR_SHA)
    previous, roles = parent.validate_contract(prior_path, PRIOR_SHA)
    require(same_json(previous, sealed), "Authenticated v5 contract differs from validated parent")
    row = evidence.binding({"path": roles["input_paths"]["contract"], "sha256": roles["inputs"]["contract_sha256"]})
    require(same_json(row["excluded_targets"], roles["excluded_targets"])
        and len(roles["excluded_targets"]) == 772 and not set(roles["targets"]) & set(roles["excluded_targets"]),
        "Protected exclusions changed")
    return {"schema": SCHEMA, "policy": policy(),
        "prior_v5_contract": {"path": str(prior_path), "sha256": PRIOR_SHA},
        "outer_registration": previous["registration"],
        "protocol": {"path": str(protocol), "sha256": evidence.digest(protocol)},
        "code_sha256": _codes(), "inputs": roles["inputs"], "input_paths": roles["input_paths"],
        "targets": roles["targets"], "excluded_targets": roles["excluded_targets"],
        "inner_folds": make_inner_folds(roles), "anchor_pools": make_anchor_pools(row, roles),
        "go_source_sha256": dict(preparation.GO_EXPECTED), "go_acquisition_path": roles["go_acquisition_path"],
        "expression_decoded": False, "response_model_fitting_performed": False,
        "prior_artifacts_modified": False, "outer_held_used_for_selection": False}, roles


def register(prior_v5_contract, protocol, outdir):
    outdir = _canonical(outdir)
    if os.path.lexists(outdir):
        raise FileExistsError("Fresh nested registration required")
    plan, _ = _plan(prior_v5_contract, protocol)
    outdir.mkdir(mode=0o700)
    plan_path = outdir / "fold_plan.json"
    _dump(plan_path, plan)  # MUST precede public GO source parsing.
    plan_sha = evidence.digest(plan_path)
    index = preparation._go_index(Path(plan["go_acquisition_path"]))
    (outdir / "go").mkdir(mode=0o700)
    folds = {}
    for fold_id, fold in plan["inner_folds"].items():
        feature_fold_id = f"{SCHEMA}:{fold_id}:{plan_sha}"
        result = go.build_features(index, target_ids=fold["outer_fit_targets"], train_target_ids=fold["fit_targets"],
            fold_id=feature_fold_id, sources=dict(preparation.GO_EXPECTED))
        destination = outdir / "go" / fold_id
        receipt = go.write_artifacts(result, destination)
        coverage = preparation.coverage_audit(result, fold["fit_targets"], fold["tuning_targets"])
        _dump(destination / "coverage.json", coverage)
        artifacts = {"feature_fold_id": feature_fold_id}
        for name, filename in (("npz", "features.npz"), ("receipt", "receipt.json"), ("coverage", "coverage.json")):
            artifacts[name + "_path"] = str(destination / filename)
            artifacts[name + "_sha256"] = evidence.digest(destination / filename)
        require(artifacts["npz_sha256"] == receipt["npz_sha256"], "Written GO artifact hash changed")
        folds[fold_id] = {**fold, "go": artifacts}
    current, _ = _plan(prior_v5_contract, protocol)
    require(same_json(current, plan), "Metadata/code changed while constructing GO artifacts")
    record = {**plan, "inner_folds": folds, "fold_plan_path": str(plan_path), "fold_plan_sha256": plan_sha,
        "inner_go_completed": True, "outer_vocabulary_reused": False}
    path = outdir / "role_manifest.json"
    _dump(path, record)
    digest = evidence.digest(path)
    _dump(outdir / "complete.json", {"schema": SCHEMA, "completed": True,
        "role_manifest_sha256": digest, "fold_plan_sha256": plan_sha,
        "inner_folds": 4, "source_directions": 8, "anchor_pools": len(plan["anchor_pools"]),
        "expression_decoded": False, "response_model_fitted": False, "prior_artifacts_modified": False})
    return digest


def validate_registration(path, sha):
    path = _canonical(path)
    record = evidence.read_json(path, sha)
    require(record.get("inner_go_completed") is True and record.get("outer_vocabulary_reused") is False,
        "New inner-only GO vocabulary required")
    require(record.get("fold_plan_path") == str(path.parent / "fold_plan.json"), "Plan path outside registration")
    plan = evidence.read_json(record["fold_plan_path"], record["fold_plan_sha256"])
    require(plan["prior_v5_contract"]["sha256"] == PRIOR_SHA, "Only frozen v5 parent admitted")
    expected, roles = _plan(Path(plan["prior_v5_contract"]["path"]), Path(plan["protocol"]["path"]))
    require(same_json(plan, expected), "Frozen nested metadata, code or protocol differs")
    extra = {"fold_plan_path", "fold_plan_sha256", "inner_go_completed", "outer_vocabulary_reused"}
    require(set(record) == set(plan) | extra and all(same_json(record.get(k), v) for k, v in plan.items() if k != "inner_folds"),
        "Manifest differs from frozen metadata plan")
    require(isinstance(record.get("inner_folds"), dict) and set(record["inner_folds"]) == set(plan["inner_folds"]),
        "Exactly four inner folds required")
    for fold_id, wanted in plan["inner_folds"].items():
        actual = record["inner_folds"][fold_id]
        require(set(actual) == set(wanted) | {"go"} and all(same_json(actual.get(k), v) for k, v in wanted.items()),
            "Inner target/source group roles differ")
        feature = actual["go"]
        require(set(feature) == {"feature_fold_id", "npz_path", "npz_sha256", "receipt_path", "receipt_sha256", "coverage_path", "coverage_sha256"},
            "Unexpected inner GO binding fields")
        require(feature["feature_fold_id"] == f"{SCHEMA}:{fold_id}:{record['fold_plan_sha256']}", "Inner GO fold provenance differs")
        documents = {}
        for name, filename in (("npz", "features.npz"), ("receipt", "receipt.json"), ("coverage", "coverage.json")):
            destination = path.parent / "go" / fold_id / filename
            require(feature[name + "_path"] == str(destination), "Inner GO artifact outside registered directory")
            documents[name] = evidence.binding({"path": str(destination), "sha256": feature[name + "_sha256"]}, json_document=name != "npz")
        receipt = documents["receipt"]
        require(receipt.get("schema") == "vcc-go-target-features-v1"
            and receipt.get("status") == "built_features_only"
            and receipt.get("expression_read") is False and receipt.get("response_model_trained") is False
            and receipt.get("policy", {}).get("include_iea") is False
            and receipt.get("policy", {}).get("ancestor_policy") == "direct_only"
            and type(receipt.get("target_count")) is int and receipt["target_count"] == 42
            and type(receipt.get("train_target_count")) is int and receipt["train_target_count"] == 28
            and receipt.get("fold_id") == feature["feature_fold_id"]
            and same_json(receipt.get("sources"), preparation.GO_EXPECTED)
            and receipt.get("train_target_ids_sha256") == go._sha_json(wanted["fit_targets"])
            and receipt.get("target_ids_sha256") == go._sha_json(wanted["outer_fit_targets"])
            and receipt.get("vocabulary_fit_scope") == "explicit_training_targets_only"
            and receipt.get("npz_sha256") == feature["npz_sha256"], "Inner GO receipt does not bind28 fit and42 output targets")
        coverage = documents["coverage"]
        require(coverage.get("fit_vocabulary_only") is True and coverage.get("expression_or_outcomes_used") is False
            and coverage.get("selection_or_filtering_performed") is False, "Invalid expression-free GO coverage receipt")
    complete = evidence.read_json(path.with_name("complete.json"))
    wanted_complete = {"schema": SCHEMA, "completed": True, "role_manifest_sha256": sha,
        "fold_plan_sha256": record["fold_plan_sha256"], "inner_folds": 4, "source_directions": 8,
        "anchor_pools": len(record["anchor_pools"]), "expression_decoded": False,
        "response_model_fitted": False, "prior_artifacts_modified": False}
    require(same_json(complete, wanted_complete), "Nested completion binding differs")
    return record, roles


def load_feature_tables(record):
    """Decode only NEW authenticated GO feature NPZs, not cell expressions."""
    return {fold_id: load_go_table(Path(fold["go"]["npz_path"]), fold["go"]["npz_sha256"],
        Path(fold["go"]["receipt_path"]), fold["go"]["receipt_sha256"], fold["fit_targets"])
        for fold_id, fold in record["inner_folds"].items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-v5-contract", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        parser.exit(2, "Registration requires an approved BioHPC compute host.\n")
    digest = register(args.prior_v5_contract, args.protocol, args.output_dir)
    print(json.dumps({"role_manifest_sha256": digest, "expression_decoded": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
