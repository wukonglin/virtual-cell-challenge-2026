"""Seal metadata-only source-specific auxiliary candidates; NOT train authorization.

No X values, cache decoding, GO fitting, model fitting, network or credentials.
Every future expression consumer must obtain a separate materialization/training
contract, reauthenticate both complete sources, and reconstruct these row roles.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import os
from pathlib import Path
import resource
import socket
import time

import prepare_public_replicated_v7 as replicated
import prepare_public_expanded_v8 as expanded
import run_public_expanded_v81 as prior
import public_training_readiness as safe

SCHEMA = "public-auxiliary-candidate-registration-v9"
REPO = Path(__file__).resolve().parents[1]
V81_RELATIVE = "artifacts/public_flow/expanded_v81_20260910/execution/contract.json"
V81_SHA = "24d4581d9637cf082a028938fe425e515bf3b9a86a8cd4c6c08659419615cab8"
ROLE_SHA = "8ec29334bfb385e002f681f6b8129a47ed19f7df3d8ed7df31cb9185503dc7d0"
EXPECTED_TARGETS = {"replogle_k562": 374, "nadig_jurkat": 113}
EXPECTED_GROUPS = {"replogle_k562": 3689, "nadig_jurkat": 1386}
EXPECTED_TREATED = {"replogle_k562": 38928, "nadig_jurkat": 18279}
EXPECTED_POOLS = {"replogle_k562": 48, "nadig_jurkat": 53}
MAX_JSON_BYTES = 16 << 20
MAX_RSS = 4 << 30
MAX_SECONDS = 1800
MODULES = tuple(dict.fromkeys(("prepare_public_auxiliary_v9.py", *prior.MODULES)))
equal = replicated.equal
require = replicated.require
canonical_path = replicated.canonical_path
old = replicated.old


def policy():
    return {"stage": "metadata_only_candidate_registration", "candidate_plan_only": True,
        "expression_materialization_authorized": False, "training_authorized": False,
        "separate_materialization_and_training_contract_required": True,
        "source_routing": "K562 auxiliaries only for K562-fit/Jurkat-held; Jurkat auxiliaries only for Jurkat-fit/K562-held",
        "shared_cross_source_treated_warmstart_allowed": False,
        "target_exclusions": "all frozen56 panel targets plus all772 protected targets, globally",
        "min_treated_per_batch": 8, "min_controls_per_batch": 72,
        "min_batches_per_target_source": 2, "max_treated_per_group": 32,
        "max_batches_per_target_source": None, "seed": 20260909,
        "row_ranking": "unchanged v2 source/batch/cell-identity SHA256 ranking",
        "identity": "canonical JSON SHA256 of namespace/source/target/batch; never concatenated group index",
        "control_reference": "frozen source/batch global fit_reference16 only",
        "control_context": "frozen disjoint context32, conditioning donors only; never response/prior targets",
        "tuning_anchor_and_sham_rows_in_candidate_plan": False,
        "panel_outer_inner_roles": "preserve frozen v8.1 outer42/14 and nested28/14; independent source warmstart cloned per fold",
        "learned_state": "encoders, transforms, priors, vocabulary, optimizer and early stopping must obey later registered source/target roles",
        "go_features_fitted_or_reused": False, "architecture_registered": False,
        "full_source_hash_performed_now": False,
        "full_source_hash_and_metadata_replay_before_any_future_X_read": True,
        "future_cache_layout": "deduplicate original source row identities and control pools; no repeated-pool cache assumption",
        "gene_axis": "unchanged7563 measured/512 output symbols; metadata only",
        "exposure": "previously exposed public development sources; not an untouched test",
        "raw_expression_values_read": False, "parent_cache_decoded": False,
        "model_fitting_performed": False, "posttraining_performed": False,
        "automatic_promotion": False, "download_performed": False,
        "submission_performed": False, "cpu_threads_max": 2,
        "peak_rss_limit_bytes": MAX_RSS, "elapsed_limit_seconds": MAX_SECONDS,
        "artifact_json_limit_bytes": MAX_JSON_BYTES,
        "runtime_limits": "RSS and elapsed checkpoints; CLI numerical thread preflight"}


def _codes():
    return {name: safe.digest(Path(__file__).with_name(name)) for name in MODULES}


def _parents(repo):
    repo = Path(repo).resolve()
    require(repo == REPO, "Frozen source lineage belongs to this exact repository")
    binding = {"path": str(repo / V81_RELATIVE), "sha256": V81_SHA}
    execution, record, roles, nested = prior.validate_contract(Path(binding["path"]), V81_SHA)
    equal(execution["registration"]["sha256"], ROLE_SHA, "frozen v8.1 roles")
    row = safe.binding(record["parents"]["v7"]["contract"])
    manifest = safe.binding(record["parents"]["v7"]["source_manifest"])
    equal(record["inputs"]["contract_sha256"], expanded.PINS["contract"][1], "frozen v7 row contract")
    equal(record["inputs"]["source_manifest_sha256"], expanded.PINS["source_manifest"][1], "frozen v7 sources")
    return {"v81_execution": binding, "v81_registration": execution["registration"],
        "v7": record["parents"]["v7"]}, row, manifest, roles, nested


def identity(kind, source, *parts):
    require(kind in {"group", "pool"} and source in replicated.SOURCE_IDS,
        "Invalid source-local candidate identity")
    require(all(isinstance(p, str) and p and p == p.strip() for p in parts),
        "Nonempty exact candidate identity required")
    require(len(parts) == (2 if kind == "group" else 1), "Wrong candidate identity dimensions")
    return hashlib.sha256(old.canonical([SCHEMA, kind, source, *parts])).hexdigest()


def _names(values, count, label):
    require(isinstance(values, list) and len(values) == count
        and all(isinstance(v, str) and v and v == v.strip() for v in values)
        and values == sorted(set(values)), "Invalid frozen " + label)
    return set(values)


def select_candidates(metas, row, manifest, roles, nested):
    """Pure metadata reconstruction; all source/anchor boundaries remain explicit."""
    panel = _names(row["common_targets"], 56, "panel")
    denied = _names(row["excluded_targets"], 772, "protected exclusions")
    require(not panel & denied, "Panel intersects protected exclusions")
    for parent in (roles, nested):
        equal(parent["targets"], row["common_targets"], "frozen panel roles")
        equal(parent["excluded_targets"], row["excluded_targets"], "frozen excluded roles")
    require(set(metas) == set(replicated.SOURCE_IDS), "Exactly two original metadata sources required")
    specs = manifest["sources"]
    require([s["id"] for s in specs] == list(replicated.SOURCE_IDS), "Original source order required")
    require([s["id"] for s in row["sources"]] == list(replicated.SOURCE_IDS), "Original row source order required")
    expected_pools = expanded.nested_preparation.make_anchor_pools(row, roles)
    equal(nested["anchor_pools"], expected_pools, "frozen global16/16 anchor roles")
    pools = {(p["source"], p["batch"]): p for p in expected_pools}
    equal(dict(Counter(p["source"] for p in expected_pools)), EXPECTED_POOLS, "frozen source control pools")
    eligible = {}
    for spec, original in zip(specs, row["sources"]):
        source, meta = spec["id"], metas[spec["id"]]
        equal({k: original[k] for k in spec}, spec, "unchanged source acquisition")
        equal(old.metadata_digest(meta), original["metadata_sha256"], "unchanged source metadata")
        require(len(meta["cells"]) == original["cells_total"]
            and len(meta["genes"]) == original["genes_total"], "Source metadata dimensions changed")
        eligible[source] = replicated.v2.eligible_batches(meta, denied, spec["control_label"])
    equal(sorted(set.intersection(*(set(eligible[s]) for s in replicated.SOURCE_IDS))),
        row["common_targets"], "unchanged eligible common panel")
    sources, all_targets = [], []
    for spec, original in zip(specs, row["sources"]):
        source, meta = spec["id"], metas[spec["id"]]
        targets = sorted(set(eligible[source]) - panel)
        require(not set(targets) & denied, "Protected auxiliary target admitted")
        require(len(targets) == EXPECTED_TARGETS[source], "Unexpected source auxiliary target count")
        all_targets.extend(targets)
        positions = {}
        admitted_labels = set(targets) | {spec["control_label"]}
        for number, (target, batch) in enumerate(zip(meta["targets"], meta["batches"])):
            if target in admitted_labels:
                positions.setdefault((str(target), str(batch)), []).append(number)
        groups, used_pools, treated_seen = [], {}, set()
        # Sorted semantic identities have no dependency on a global roster index.
        for target in targets:
            for batch in sorted(eligible[source][target]):
                require((source, batch) in pools, "Auxiliary batch lacks frozen panel control roles")
                p = pools[(source, batch)]
                pool_id = identity("pool", source, batch)
                if batch not in used_pools:
                    ranked = replicated.v2.ranked_rows(meta, positions[(spec["control_label"], batch)], source, batch)
                    equal(p["context_rows"], sorted(ranked[:32]), "metadata context control selection")
                    equal(sorted(p["fit_reference_rows"] + p["tuning_anchor_rows"]), sorted(ranked[32:64]),
                        "metadata anchor control selection")
                    equal(p["sham_rows"], sorted(ranked[64:72]), "metadata sham control selection")
                    used_pools[batch] = {"pool_id": pool_id, "source": source, "batch": batch,
                        "fit_reference_rows": list(p["fit_reference_rows"]), "context_rows": list(p["context_rows"])}
                rows = sorted(replicated.v2.ranked_rows(meta, positions[(target, batch)], source, batch)[:32])
                require(8 <= len(rows) <= 32 and not treated_seen & set(rows), "Repeated/undersized treated candidates")
                forbidden = set(p["context_rows"] + p["fit_reference_rows"] + p["tuning_anchor_rows"] + p["sham_rows"])
                require(not set(rows) & forbidden, "Treated candidate overlaps a control role")
                treated_seen.update(rows)
                groups.append({"group_id": identity("group", source, target, batch),
                    "source": source, "target": target, "batch": batch,
                    "pool_id": pool_id, "treated_rows": rows})
        require(len(groups) == EXPECTED_GROUPS[source] and len(treated_seen) == EXPECTED_TREATED[source],
            "Unexpected auxiliary group/treated capacity")
        require(len(used_pools) == EXPECTED_POOLS[source], "Unexpected auxiliary control pool coverage")
        sources.append({"id": source, "acquisition": copy.deepcopy(spec),
            "metadata_sha256": original["metadata_sha256"], "targets": targets,
            "groups": groups, "control_pools": [used_pools[b] for b in sorted(used_pools)]})
    require(len(all_targets) == len(set(all_targets)) == sum(EXPECTED_TARGETS.values()),
        "Auxiliary targets overlap source-specific eligibility cohorts")
    return sources


def capacity(sources):
    result = {}
    for source in sources:
        treated = {r for g in source["groups"] for r in g["treated_rows"]}
        reference = {r for p in source["control_pools"] for r in p["fit_reference_rows"]}
        context = {r for p in source["control_pools"] for r in p["context_rows"]}
        require(not treated & reference and not treated & context and not reference & context,
            "Global candidate treated/reference/context role overlap")
        result[source["id"]] = {"targets": len(source["targets"]), "groups": len(source["groups"]),
            "treated_rows": len(treated), "fit_reference_rows": len(reference),
            "context_rows": len(context), "control_pools": len(source["control_pools"]),
            "unique_candidate_rows": len(treated | reference | context),
            "deduplicated512_float32_expression_bytes_if_later_authorized": len(treated | reference | context) * 512 * 4}
    return result


def routing(sources, roles):
    output = []
    for source in sources:
        source_id = source["id"]
        held = next(s for s in replicated.SOURCE_IDS if s != source_id)
        folds = []
        for fold in roles["folds"]:
            matches = [d for d in fold["directions"] if d["fit_source"] == source_id and d["held_source"] == held]
            require(len(matches) == 1, "Missing frozen source direction")
            folds.append({"outer_fold_id": fold["fold_id"], "direction_id": matches[0]["direction_id"]})
        require(len(folds) == 4, "Four frozen fold clones required")
        output.append({"fit_source": source_id, "held_source": held, "source_specific_warmstart_only": True,
            "candidate_group_ids": [g["group_id"] for g in source["groups"]], "clone_per_outer_fold": folds,
            "held_source_treated_used": False, "training_authorized": False})
    return output


def _build(parents, protocol, metas):
    bindings, row, manifest, roles, nested = parents
    sources = select_candidates(metas, row, manifest, roles, nested)
    return {"schema": SCHEMA, "policy": policy(), "parents": bindings,
        "protocol": {"path": str(protocol), "sha256": safe.digest(protocol)}, "code_sha256": _codes(),
        "panel_targets": row["common_targets"], "protected_excluded_targets": row["excluded_targets"],
        "shared_genes": row["shared_genes"], "output_genes": row["output_genes"],
        "sources": sources, "source_routing": routing(sources, roles), "capacity": capacity(sources),
        "candidate_groups": sum(EXPECTED_GROUPS.values()), "candidate_targets": sum(EXPECTED_TARGETS.values()),
        "candidate_treated_rows": sum(EXPECTED_TREATED.values()),
        "raw_expression_values_read": False, "parent_cache_decoded": False,
        "full_source_hash_performed_now": False, "expression_materialization_authorized": False,
        "training_authorized": False, "go_feature_artifacts_registered": False,
        "architecture_registered": False, "posttraining_performed": False,
        "automatic_promotion": False, "submission_performed": False}


def _completion(sha, record):
    return {"schema": SCHEMA, "completed": True, "candidate_plan_sha256": sha,
        "candidate_groups": record["candidate_groups"], "candidate_targets": record["candidate_targets"],
        "candidate_treated_rows": record["candidate_treated_rows"], "metadata_only": True,
        "expression_materialization_authorized": False, "training_authorized": False}


def _checkpoint(started):
    require(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 <= MAX_RSS,
        "Metadata registration exceeded4GiB peak RSS")
    require(time.monotonic() - started <= MAX_SECONDS, "Metadata registration exceeded30min")


def _dump(path, record):
    require(len(old.canonical(record)) <= MAX_JSON_BYTES, "Oversized candidate registration JSON")
    replicated.dump_new(path, record)


def register(repo, output_dir, protocol_path):
    started = time.monotonic()
    output_dir, protocol = canonical_path(output_dir), canonical_path(protocol_path)
    if os.path.lexists(output_dir):
        raise FileExistsError("Fresh candidate registration directory required")
    parents = _parents(repo)
    metas = replicated.read_metadata_sources(parents[2]["sources"])
    record = _build(parents, protocol, metas)
    _checkpoint(started)
    # Reauthenticate frozen parents, code and protocol before publishing a seal.
    equal(_build(_parents(repo), protocol, metas), record, "candidate parent/code/protocol recheck")
    _checkpoint(started)
    output_dir.mkdir(mode=0o700)
    path = output_dir / "candidate_plan.json"
    _dump(path, record)
    sha = safe.digest(path)
    _dump(output_dir / "complete.json", _completion(sha, record))
    return {"candidate_plan_sha256": sha, **_completion(sha, record), "capacity": record["capacity"]}


def validate_metadata(record, metas, *, repo=REPO):
    require(isinstance(record, dict) and record.get("schema") == SCHEMA, "Wrong candidate registration schema")
    protocol = canonical_path(record["protocol"]["path"])
    equal(record, _build(_parents(repo), protocol, metas), "complete candidate metadata reconstruction")
    return record


def load_registration(path, sha, *, repo=REPO):
    started = time.monotonic()
    path = canonical_path(path)
    require(path.name == "candidate_plan.json", "Canonical candidate_plan.json leaf required")
    record = safe.read_json(path, sha)
    parents = _parents(repo)
    metas = replicated.read_metadata_sources(parents[2]["sources"])
    equal(record, _build(parents, canonical_path(record["protocol"]["path"]), metas),
        "complete candidate registration replay")
    equal(safe.read_json(path.with_name("complete.json")), _completion(sha, record), "candidate completion binding")
    _checkpoint(started)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    args = parser.parse_args(argv)
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"},
        "Metadata registration requires an approved compute host, never a login node")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Candidate registration does not use GPUs")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        require(os.environ.get(name) == "2", "Set numerical thread limits to two")
    print(old.canonical(register(args.repo, args.output_dir, args.protocol)).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
