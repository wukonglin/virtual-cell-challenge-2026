"""Expression-free expanded-cohort roles with unchanged public target splits.

The completed v7 cache is authenticated as opaque bytes, never decoded here.
The original v3 outer and v6 inner GO artifacts are reused because their exact
target fitting/output roles are unchanged. No raw sources, credentials, network,
model fitting, post-training or submissions are opened by registration.
"""
from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import socket

import prepare_public_joint_target_source_v3 as outer_preparation
import prepare_public_nested_v6 as nested_preparation
import prepare_public_replicated_v7 as replicated
import public_training_readiness as safe
import run_public_nested_v6 as previous

SCHEMA = "public-expanded-target-reference-registration-v8"
OUTER_SCHEMA = "public-expanded-outer-target-source-roles-v8"
NESTED_SCHEMA = "public-expanded-inner-target-reference-roles-v8"
REPO = Path(__file__).resolve().parents[1]
V7_RELATIVE = "artifacts/public_flow/replicated_v7_20260910"
PINS = {
    "contract": ("registration/contract.json", "c2fe975be373f3c7eb6be20488a0307ebd4f7f355d87b5672977f814b328f644"),
    "source_manifest": ("registration/sources.json", "df6fc7a7a7da6d4ba2a241c0a73849a7aeb15a0f42c71de6cd69d72f288d3516"),
    "cache": ("cache.npz", "292e52e7e2818e754f8c9176ff17d9b1542c07d7f1de2a43ed40a0912132f783"),
    "cache_receipt": ("cache.npz.receipt.json", "a3cd83308ab7fa8fd45bdc5a4de5acce4768e7c7d1134919724e24071261ad0e"),
}
V6_RELATIVE = replicated.V6_RELATIVE
V6_SHA = replicated.V6_SHA
V6_ROLE_SHA = "4e5d87a1790f2e5062cb735fef66d1dcf9d8fb83e31366f872d6d2720ebd2d85"
V3_ROLE_SHA = "53954afe6a04c4bf9f6ff35fef9b8d25c5c91c6da434e70cc629c5e8dbe9bf34"
EXPECTED_GROUPS = {"replogle_k562": 953, "nadig_jurkat": 814}
EXPECTED_POOLS = 101
EXPECTED_PARENT_POOLS = 97
MODULES = tuple(dict.fromkeys(("prepare_public_expanded_v8.py", *replicated.DEPENDENCIES)))
equal = replicated.equal
require = replicated.require
canonical_path = replicated.canonical_path
dump_new = replicated.dump_new


def policy():
    return {"schema": SCHEMA, "stage": "expanded_public_nested_validation",
        "sampling": "completed v7 replication expansion; no additional sampling or expression normalization",
        "target_roles": "unchanged v3 four42/14 outer folds and v6 nested28/14 splits",
        "outer_sources": "fit one source; evaluate opposite source and held targets",
        "inner_sources": "fit and tune within outer fitting source only",
        "inner_target_rank": "reuse frozen v6 seed20260925 and nested-target-v6 JSON-array SHA256 rank",
        "anchor_rank": "reuse frozen v6 seed20260925 and nested-anchor-v6 JSON-array SHA256 rank",
        "anchor_scope": "one globally disjoint16/16 original-row split per source/batch;101 pools",
        "context_use": "disjoint32 control-derived context donors; never fitting response or positive prior",
        "go_reuse": "reuse authenticated v3 outer42-fit and v6 inner28-fit GO artifacts; target roles unchanged",
        "new_go_vocabulary_fitted": False, "full56_vocabulary_used_for_fitting": False,
        "exposure": "previously exposed public development sources; not an untouched test",
        "excluded_expression": "RPE1/H1/HepG2/Feng/challenge2026 excluded; preserve772 target exclusions",
        "cpu_threads_max": 8, "peak_rss_limit_bytes": 16 << 30,
        "artifact_limit_bytes": 32 << 30, "per_npz_limit_bytes": 512 << 20,
        "artifact_layout": "separate outer-fold and inner-candidate shards; no whole-arm prediction archive",
        "expression_decoded_during_registration": False, "raw_sources_opened": False,
        "response_model_fitting_performed": False, "posttraining_performed": False,
        "automatic_promotion": False, "submission_performed": False}


def _codes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in MODULES}


def _parents():
    """Authenticate all parent metadata and opaque cache/GO bytes, never X."""
    bindings = {name: {"path": str(REPO / V7_RELATIVE / filename), "sha256": sha}
        for name, (filename, sha) in PINS.items()}
    row = replicated.load_contract(Path(bindings["contract"]["path"]), bindings["contract"]["sha256"],
        Path(bindings["source_manifest"]["path"]), bindings["source_manifest"]["sha256"], repo=REPO)
    safe.binding(bindings["cache"], json_document=False)
    receipt = safe.binding(bindings["cache_receipt"])
    for key in ("cache", "contract", "source_manifest"):
        equal(receipt.get(key + "_sha256"), bindings[key]["sha256"], "v7 receipt " + key)
    equal(receipt.get("cache_path"), bindings["cache"]["path"], "v7 receipt cache path")
    expected = {"schema": "public-replicated-control-anchor-cache-receipt-v7", "completed": True,
        "groups": sum(EXPECTED_GROUPS.values()), "stage": "data_preparation",
        "model_fitting_performed": False, "posttraining_performed": False, "submission_performed": False,
        "rpe1_h1_hepg2_challenge2026_feng_opened": False}
    for key, value in expected.items():
        equal(receipt.get(key), value, "v7 receipt " + key)
    execution_binding = {"path": str(REPO / V6_RELATIVE), "sha256": V6_SHA}
    execution, outer, nested = previous.validate_contract(Path(execution_binding["path"]), V6_SHA)
    equal(execution["nested_registration"]["sha256"], V6_ROLE_SHA, "frozen v6 roles")
    equal(execution["registration"]["sha256"], V3_ROLE_SHA, "frozen v3 roles")
    parents = {"v7": bindings, "v6_execution": execution_binding,
        "v6_nested_registration": execution["nested_registration"],
        "v3_outer_registration": execution["registration"]}
    return parents, row, outer, nested


def make_roles(row, outer_parent, nested_parent, inputs, input_paths):
    """Reconstruct every expanded role; copy only authenticated GO bindings."""
    targets = nested_preparation._names(row["common_targets"], 56, "expanded targets")
    excluded = nested_preparation._names(row["excluded_targets"], 772, "protected exclusions")
    require(not set(targets) & set(excluded), "Excluded target admitted")
    for parent in (outer_parent, nested_parent):
        equal(parent["targets"], targets, "parent target cohort")
        equal(parent["excluded_targets"], excluded, "parent protected exclusions")
    sources = row["sources"]
    require(isinstance(sources, list) and [s["id"] for s in sources] == list(replicated.SOURCE_IDS),
        "Exact ordered public source pair required")
    equal({s["id"]: len(s["groups"]) for s in sources}, EXPECTED_GROUPS, "expanded source group counts")
    roster = replicated.group_roster(sources)
    equal(row["group_roster"], roster, "v7 group roster")
    require(all(g["target"] in targets and isinstance(g["batch"], str) and g["batch"] == g["batch"].strip()
        and bool(g["batch"]) for g in roster), "Invalid expanded source/target/batch")
    require(len({(g["source"], g["target"], g["batch"]) for g in roster}) == len(roster),
        "Repeated expanded source/target/batch")
    parent_folds = outer_preparation.make_folds(targets, outer_parent["group_roster"])
    require(isinstance(outer_parent["folds"], list) and len(outer_parent["folds"]) == 4,
        "Exactly four frozen outer folds required")
    for actual, wanted in zip(outer_parent["folds"], parent_folds):
        equal({k: v for k, v in actual.items() if k != "go"}, wanted, "frozen outer roles")
        require("go" in actual, "Frozen outer GO binding required")
    folds = outer_preparation.make_folds(targets, roster)
    for fold, parent in zip(folds, outer_parent["folds"]):
        for name in ("fold_id", "fit_targets", "held_targets"):
            equal(fold[name], parent[name], "unchanged outer " + name)
        fold["go"] = copy.deepcopy(parent["go"])
    roles = {"schema": OUTER_SCHEMA, "targets": targets, "excluded_targets": excluded,
        "inputs": inputs, "input_paths": input_paths, "group_roster": roster, "folds": folds,
        "go_source_sha256": copy.deepcopy(outer_parent["go_source_sha256"]),
        "go_acquisition_path": outer_parent["go_acquisition_path"], "frozen_outer_go_reused": True}
    inner_parent = nested_preparation.make_inner_folds(outer_parent)
    equal(len(nested_parent["inner_folds"]), 4, "frozen inner fold count")
    require(set(nested_parent["inner_folds"]) == set(inner_parent), "Frozen inner fold IDs differ")
    for key, wanted in inner_parent.items():
        actual = nested_parent["inner_folds"][key]
        equal({k: v for k, v in actual.items() if k != "go"}, wanted, "frozen inner roles")
        require("go" in actual, "Frozen inner GO binding required")
    inner = nested_preparation.make_inner_folds(roles)
    for key, fold in inner.items():
        for name in ("outer_fold_id", "fit_targets", "tuning_targets", "outer_fit_targets", "outer_held_targets"):
            equal(fold[name], nested_parent["inner_folds"][key][name], "unchanged inner " + name)
        fold["go"] = copy.deepcopy(nested_parent["inner_folds"][key]["go"])
    pools = nested_preparation.make_anchor_pools(row, roles)
    require(len(pools) == EXPECTED_POOLS, "Exactly101 expanded source/batch pools required")
    by_pool = {(p["source"], p["batch"]): p for p in pools}
    previous_pools = nested_parent["anchor_pools"]
    require(len(previous_pools) == EXPECTED_PARENT_POOLS, "Exactly97 frozen source/batch pools required")
    require(len({(p["source"], p["batch"]) for p in previous_pools}) == len(previous_pools),
        "Duplicate frozen source/batch pool")
    for pool in previous_pools:
        equal(by_pool.get((pool["source"], pool["batch"])), pool, "preserved old globally disjoint references")
    mapping = row["parent_group_mapping"]
    require(isinstance(mapping, list) and len(mapping) == len(outer_parent["group_roster"]),
        "Complete semantic parent mapping required")
    for number, (mapped, old) in enumerate(zip(mapping, outer_parent["group_roster"])):
        require(isinstance(mapped, dict) and set(mapped) == {"parent_group_id", "expanded_group_id"}
            and type(mapped["parent_group_id"]) is int and mapped["parent_group_id"] == number
            and type(mapped["expanded_group_id"]) is int and 0 <= mapped["expanded_group_id"] < len(roster),
            "Invalid typed semantic parent mapping")
        new = roster[mapped["expanded_group_id"]]
        equal({k: v for k, v in old.items() if k != "group_id"},
            {k: v for k, v in new.items() if k != "group_id"}, "semantic parent identity")
    require(len({m["expanded_group_id"] for m in mapping}) == len(mapping), "Repeated semantic parent group")
    nested = {"schema": NESTED_SCHEMA, "targets": targets, "excluded_targets": excluded,
        "inputs": inputs, "input_paths": input_paths, "inner_folds": inner, "anchor_pools": pools,
        "frozen_inner_go_reused": True, "outer_response_selection": False,
        "expression_decoded": False, "response_model_fitting_performed": False}
    return roles, nested


def _plan(protocol):
    protocol = canonical_path(protocol)
    parents, row, outer, nested = _parents()
    inputs = {key + "_sha256": parents["v7"][key]["sha256"] for key in ("cache", "contract", "source_manifest")}
    paths = {key: parents["v7"][key]["path"] for key in ("cache", "contract", "source_manifest")}
    roles, expanded_nested = make_roles(row, outer, nested, inputs, paths)
    return {"schema": SCHEMA, "policy": policy(), "parents": parents, "inputs": inputs, "input_paths": paths,
        "protocol": {"path": str(protocol), "sha256": safe.digest(protocol)}, "code_sha256": _codes(),
        "roles": roles, "nested": expanded_nested, "groups": len(roles["group_roster"]),
        "anchor_pools": len(expanded_nested["anchor_pools"]),
        "parent_groups_preserved": len(row["parent_group_mapping"]),
        "outer_go_reused": True, "inner_go_reused": True,
        "new_go_vocabulary_fitted": False, "expression_decoded": False,
        "response_model_fitting_performed": False, "prior_artifacts_modified": False}


def _completion(sha, record):
    return {"schema": SCHEMA, "completed": True, "role_manifest_sha256": sha,
        "groups": record["groups"], "anchor_pools": record["anchor_pools"], "outer_folds": 4,
        "source_directions": 8, "inner_folds": 4, "expression_decoded": False,
        "response_model_fitting_performed": False, "prior_artifacts_modified": False}


def register(protocol: Path, output_dir: Path):
    output_dir = canonical_path(output_dir)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh expanded registration directory required")
    record = _plan(protocol)
    output_dir.mkdir(mode=0o700)
    path = output_dir / "role_manifest.json"
    dump_new(path, record)
    sha = safe.digest(path)
    # Fail without a completion marker if code, metadata or opaque cache changed.
    equal(_plan(protocol), record, "expanded registration recheck")
    dump_new(output_dir / "complete.json", _completion(sha, record))
    return sha


def validate_registration(path: Path, sha: str):
    path = canonical_path(path)
    require(path.name == "role_manifest.json", "Canonical role_manifest.json leaf required")
    record = safe.read_json(path, sha)
    expected = _plan(Path(record["protocol"]["path"]))
    equal(record, expected, "expanded role registration")
    equal(safe.read_json(path.with_name("complete.json")), _completion(sha, record), "expanded completion")
    return record, copy.deepcopy(record["roles"]), copy.deepcopy(record["nested"])


def load_feature_tables(record):
    """Reauthenticate unchanged parent roles before decoding public GO only."""
    equal(record, _plan(Path(record["protocol"]["path"])), "expanded feature registration")
    _, _, outer, nested = _parents()
    return outer_preparation.load_feature_tables(outer), nested_preparation.load_feature_tables(nested)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3", "cbsuvlaminck6"}:
        parser.exit(2, "Expanded registration requires an approved BioHPC compute host.\n")
    digest = register(args.protocol, args.output_dir)
    print(replicated.json_bytes({"role_manifest_sha256": digest, "expression_decoded": False}).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
