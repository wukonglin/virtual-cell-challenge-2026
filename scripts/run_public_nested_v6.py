"""Register and run public-only nested regularization; no promotion or upload."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import numpy as np

import prepare_public_nested_v6 as preparation
import prepare_public_joint_target_source_v3 as outer_preparation
import public_nested_v6 as model
import public_joint_target_source_v3 as folds_module
import run_public_target_context_v5 as prior
import public_training_readiness as safe
from public_nested_tracking_v6 import METRICS, NestedValidationTracker
from prepare_public_control_anchor_v2 import load_cache
from train_public_flow_pilot import write_new
from wandb_training import finite_scalar

SCHEMA="public-nested-regularization-execution-v6"
SOURCES={"nadig_jurkat","replogle_k562"}
MODULES=tuple(dict.fromkeys(("run_public_nested_v6.py","public_nested_v6.py","prepare_public_nested_v6.py",
    "public_target_context_kernel_v6.py","public_nested_metrics_v6.py","public_nested_tracking_v6.py",
    "public_nested_readiness_v6.py",*prior.MODULES)))
FLAGS_FALSE=("count_emitter","posttraining_performed","promoted","submission_performed",
    "H1_treated_read","RPE1_treated_read","challenge_treated_used","outer_evaluation_roles_used_for_fitting","outer_response_selection")
AUDIT_TRUE=("passed","saved_head_replay","training_prior_replay","kernel_system_verified","training_only_transforms_verified",
    "deterministic_donor_replay","original_row_mapping_verified","pooled_and_batch_metrics_replayed",
    "training_diagnostics_replayed","inner_selection_replayed","disjoint_inner_references_verified")
sha256_file=safe.digest
authenticated_json=safe.read_json

def equal(actual,expected,label):
    if model.canonical(actual)!=model.canonical(expected): raise ValueError("Typed "+label+" mismatch")

def dump_new(path,value):
    write_new(path,(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())

def screen_policy():
    return {"minimum_relative_primary_improvement": 0.01,
        "primary_comparators": ["control", *model.SHUFFLED_ARMS],
        "strict_context_primary_improvement": True,
        "strict_additive_primary_improvement": True,
        "batch_centroid_mse_noninferior_to_control_and_context": True,
        "mmd_noninferior_to_all_comparators": True,
        "occupancy_mse_noninferior_to_control_and_context": True,
        "aggregate_zero_rate_error_noninferior_to_control": True,
        "maximum_occupancy_clip_fraction": 0.01, "maximum_amplitude_clip_fraction": 0.01,
        "unresolved_activation_fraction": 0.0,
        "both_sources_required": sorted(SOURCES), "automatic_promotion": False,
        "compatible_parent_and_full_count_emitter_not_established": True}


def register(role_path,role_sha,protocol,output_dir):
    if os.path.lexists(output_dir): raise FileExistsError("Fresh nested execution directory required")
    nested,roles=preparation.validate_registration(role_path,role_sha)
    equal(nested["protocol"],{"path":str(protocol.resolve()),"sha256":sha256_file(protocol)},"protocol binding")
    record={"schema":SCHEMA,"created_at":datetime.now(timezone.utc).isoformat(),
        "nested_registration":{"path":str(role_path.resolve()),"sha256":role_sha},
        "registration":nested["outer_registration"],"inputs":roles["inputs"],"policy":model.policy(),"screen":screen_policy(),
        "protocol":nested["protocol"],"code_sha256":{name:sha256_file(Path(__file__).with_name(name)) for name in MODULES},
        "expression_decoded":False,"new_raw_expression_allowed":False,"automatic_promotion":False,
        "submission_allowed":False,"stage":"validation","cpu_only":True,"wandb_mode":"offline",
        "events_by_arm":{a:9 if a=="control" else 33 for a in model.ARMS},"total_events":207,
        "hyperparameter_selection_scope":"inner_only_disjoint_reference_rows","outer_response_selection":False}
    output_dir.mkdir(mode=0o700); dump_new(output_dir/"contract.json",record)
    return sha256_file(output_dir/"contract.json")

def validate_contract(path,digest):
    value=authenticated_json(path,digest)
    expected={"schema":SCHEMA,"policy":model.policy(),"screen":screen_policy(),"expression_decoded":False,
        "new_raw_expression_allowed":False,"automatic_promotion":False,"submission_allowed":False,
        "stage":"validation","cpu_only":True,"wandb_mode":"offline",
        "events_by_arm":{a:9 if a=="control" else 33 for a in model.ARMS},"total_events":207,
        "hyperparameter_selection_scope":"inner_only_disjoint_reference_rows","outer_response_selection":False}
    for key,v in expected.items(): equal(value.get(key),v,"execution "+key)
    codes=value.get("code_sha256",{})
    if set(codes)!=set(MODULES) or any(sha256_file(Path(__file__).with_name(k))!=v for k,v in codes.items()):
        raise ValueError("Nested execution code changed")
    binding=value["nested_registration"]
    nested,roles=preparation.validate_registration(Path(binding["path"]),binding["sha256"])
    equal(value["registration"],nested["outer_registration"],"outer registration")
    equal(value["inputs"],roles["inputs"],"input binding")
    equal(value["protocol"],nested["protocol"],"protocol")
    safe.binding(value["protocol"],json_document=False)
    return value,roles,nested

def verify_metrics(values):
    if not isinstance(values,dict) or not set(METRICS)<=set(values): raise ValueError("Missing metric")
    for key in METRICS:
        value=values[key]
        if not finite_scalar(value) or value<0: raise ValueError("Invalid metric")
        if key.endswith("fraction") and value>1: raise ValueError("Invalid fraction")
    for key in ("raw_model_mmd2","raw_control_mmd2"):
        if key in values and (not finite_scalar(values[key]) or values[key]<-1e-12):
            raise ValueError("Invalid raw MMD")

def expected_folds(roles):
    roster=roles["group_roster"]
    arrays={"targets":np.asarray([r["target"] for r in roster]),
        "sources":np.asarray([r["source"] for r in roster])}
    folds=folds_module.joint_folds(tuple(arrays["targets"]),tuple(arrays["sources"]),roles)
    return arrays,folds

def verify_report(report,fit_count,held_count,expected=None,roster=None):
    if not isinstance(report,dict): raise ValueError("Missing fold report")
    fit,held=report.get("fit_targets"),report.get("held_targets")
    for names,count in ((fit,fit_count),(held,held_count)):
        if (not isinstance(names,list) or len(names)!=count or any(type(t) is not str or not t for t in names)
            or names!=sorted(set(names))): raise ValueError("Unique sorted target roles required")
    if set(fit)&set(held): raise ValueError("Overlapping target roles")
    verify_metrics(report.get("metrics"))
    if expected is not None:
        meta={"fold_id":expected.fold_id,"target_fold":expected.target_fold,"fit_source":expected.fit_source,
            "held_source":expected.held_source,"fit_targets":list(expected.fit_targets),"held_targets":list(expected.held_targets),
            "fit_group_ids":list(expected.fit_indices),"joint_eval_group_ids":list(expected.evaluation_indices),
            "held_target_group_ids_all_sources":list(expected.held_target_indices_all_sources)}
        for key,v in meta.items(): equal(report.get(key),v,"fold "+key)
        groups=report.get("groups",[])
        if not isinstance(groups,list) or len(groups)!=len(expected.evaluation_indices): raise ValueError("Exact group roster required")
        for row,gid in zip(groups,expected.evaluation_indices):
            for key in ("source","target","batch"): equal(row.get(key),roster[gid][key],"group "+key)
            equal(row.get("group"),gid,"group ID")
    lam=report.get("regularization")
    if type(lam) is not float or lam not in model.GRID: raise ValueError("Invalid saved regularization")

def verify_summary(summary,arm,contract_sha=None,*,contract=None,roles=None,nested=None):
    if arm not in model.ARMS or summary.get("schema")!="public-nested-regularization-summary-v6":
        raise ValueError("Invalid nested summary schema/arm")
    expected={"arm":arm,"status":"completed_unpromoted_diagnostic","stage":"validation","policy":model.policy(),
        "prior_public_development_exposure_acknowledged":True,"exact_ntc_identity_verified":True,
        "exact_zero_effect_identity_verified":True,"hyperparameter_selection":arm!="control",
        "inner_selection_performed":arm!="control","selection_scope":"inner_only_disjoint_reference_rows",
        **dict.fromkeys(FLAGS_FALSE,False)}
    for key,v in expected.items(): equal(summary.get(key),v,"summary "+key)
    runtime=summary.get("runtime",{})
    if (runtime.get("device")!="cpu" or not isinstance(runtime.get("hostname"),str)
        or runtime["hostname"].split(".")[0] not in {"cbsuvlaminck3","cbsuvlaminck6"}):
        raise ValueError("Summary requires approved CPU runtime")
    if contract_sha is not None:
        equal(summary.get("diagnostic_contract_sha256"),contract_sha,"summary contract")
        equal(summary.get("provenance",{}).get("diagnostic_contract_sha256"),contract_sha,"provenance contract")
    if contract is not None:
        wanted={"diagnostic_contract_sha256":contract_sha,"role_manifest_sha256":contract["registration"]["sha256"],
            "nested_manifest_sha256":contract["nested_registration"]["sha256"],"inputs":contract["inputs"]}
        equal(summary.get("provenance"),wanted,"full provenance")
    verify_metrics(summary.get("aggregate"))
    rows=summary.get("source_metrics")
    if not isinstance(rows,list) or len(rows)!=2 or {r.get("source") for r in rows}!=SOURCES:
        raise ValueError("Both unique public sources required")
    for row in rows: verify_metrics(row["metrics"])
    reports=summary.get("folds",[])
    if not isinstance(reports,list) or len(reports)!=8 or len({r["fold_id"] for r in reports})!=8:
        raise ValueError("Eight distinct outer directions required")
    axes,folds=expected_folds(roles) if roles is not None else (None,[None]*8)
    for number,(report,fold) in enumerate(zip(reports,folds)):
        verify_report(report,42,14,fold,roles["group_roster"] if roles else None)
        equal(report.get("validation_kind"),"outer_joint_target_source","outer validation kind")
        equal(report.get("outer_fold_id"),"","outer parent ID")
        if report.get("fit_source") not in SOURCES or report.get("held_source") not in SOURCES or report["fit_source"]==report["held_source"]:
            raise ValueError("Invalid outer source direction")
        selection=report.get("inner_selection",{})
        candidates=selection.get("candidates")
        if arm=="control":
            equal(selection,model.selection_record(arm,[],10.0),"control no-selection")
        else:
            selected=model.select_regularization(candidates)
            equal(selection,model.selection_record(arm,candidates,selected),"inner selection")
            inner=model.inner_fold(axes,fold,nested) if fold is not None else None
            for index,candidate in enumerate(candidates):
                verify_report(candidate,28,14,inner,roles["group_roster"] if roles else None)
                equal(candidate.get("validation_kind"),"inner_same_source_target","inner kind")
                equal(candidate.get("outer_fold_id"),report["fold_id"],"inner parent")
                equal(candidate.get("fit_source"),report["fit_source"],"inner fitting source")
                equal(candidate.get("held_source"),report["fit_source"],"inner tuning source")
                equal(candidate.get("array_prefix"),f"fold_{number}_inner_{index}_","inner prefix")
        equal(report["regularization"],selection["selected_regularization"],"selected lambda")
        equal(report.get("array_prefix"),f"fold_{number}_outer_","outer prefix")

def screen(summaries):
    if set(summaries) != set(model.ARMS):
        raise ValueError("All seven interaction arms required")
    for arm, summary in summaries.items():
        verify_summary(summary, arm)
    rows = []
    for source in sorted(SOURCES):
        values = {arm: next(r["metrics"] for r in s["source_metrics"] if r["source"] == source)
                  for arm, s in summaries.items()}
        true = values["true"]
        gains = {arm: 1 - true["target_pooled_mse"] / v["target_pooled_mse"]
            if v["target_pooled_mse"] > 0 else None for arm, v in values.items() if arm != "true"}
        conditioning = {f"primary_vs_{arm}": values[arm]["target_pooled_mse"] > 0
            and true["target_pooled_mse"] <= (1 - screen_policy()["minimum_relative_primary_improvement"]) * values[arm]["target_pooled_mse"]
            for arm in ("control", *model.SHUFFLED_ARMS)}
        conditioning["primary_vs_context"] = true["target_pooled_mse"] < values["context"]["target_pooled_mse"]
        conditioning["primary_vs_additive"] = true["target_pooled_mse"] < values["additive"]["target_pooled_mse"]
        distribution = {f"mmd_vs_{arm}": true["model_mmd2"] <= v["model_mmd2"]
            for arm, v in values.items() if arm != "true"}
        for arm in ("control", "context"):
            distribution[f"occupancy_vs_{arm}"] = true["target_pooled_occupancy_mse"] <= values[arm]["target_pooled_occupancy_mse"]
            distribution[f"batch_centroid_vs_{arm}"] = true["model_centroid_mse"] <= values[arm]["model_centroid_mse"]
        distribution["aggregate_zero_error_vs_control"] = (
            abs(true["zero_fraction"] - true["treated_zero_fraction"])
            <= abs(true["control_zero_fraction"] - true["treated_zero_fraction"]))
        safety = {"nonnegative": true["negative_fraction"] == 0,
            "no_unresolved_activation": true["unresolved_activation_fraction"] == 0,
            "occupancy_clipping_bounded": true["occupancy_clip_fraction"] <= .01,
            "amplitude_clipping_bounded": true["amplitude_clip_fraction"] <= .01,
            "identities_verified": summaries["true"]["exact_ntc_identity_verified"]
                and summaries["true"]["exact_zero_effect_identity_verified"]}
        checks = {**conditioning, **distribution, **safety}
        rows.append({"source": source, "checks": checks, "relative_primary_improvement": gains,
            "conditioning_passed": all(conditioning.values()), "distribution_passed": all(distribution.values()),
            "safety_passed": all(safety.values()), "passed": all(checks.values())})
    return {"policy": screen_policy(), "sources": rows,
        **{key: all(row[key] for row in rows) for key in ("passed", "conditioning_passed", "distribution_passed", "safety_passed")},
        "automatic_promotion": False, "leaderboard_improvement_claimed": False}


def verify_audit(audit,arm):
    if not isinstance(audit,dict) or any(audit.get(k) is not True for k in AUDIT_TRUE):
        raise ValueError("Incomplete saved artifact audit")
    for k,v in {"ridge_refits":0,"inner_candidates_replayed":0 if arm=="control" else 24,
                "out_of_fold_groups":398,"out_of_fold_source_targets":112}.items():
        if type(audit.get(k)) is not int or audit[k]!=v: raise ValueError("Invalid artifact audit count")

def run(contract_path,contract_sha,output_dir):
    if socket.gethostname().split(".")[0] not in {"cbsuvlaminck3","cbsuvlaminck6"}:
        raise RuntimeError("Use approved BioHPC compute host, never a login/head node")
    contract,roles,nested=validate_contract(contract_path,contract_sha)
    if os.path.lexists(output_dir): raise FileExistsError("Fresh nested suite output required")
    output_dir.mkdir(mode=0o700)
    paths={k:Path(v) for k,v in roles["input_paths"].items()}; inputs=roles["inputs"]
    outer_tables=outer_preparation.load_feature_tables(roles); inner_tables=preparation.load_feature_tables(nested)
    arrays,row_contract=load_cache(paths["cache"],inputs["cache_sha256"],paths["contract"],inputs["contract_sha256"],
        paths["source_manifest"],inputs["source_manifest_sha256"])
    equal(row_contract["excluded_targets"],roles["excluded_targets"],"protected exclusions")
    for value in arrays.values(): value.setflags(write=False)
    summaries,runs={},[]
    for arm in model.ARMS:
        tracker=NestedValidationTracker(output_dir/"tracking"/arm,arm=arm,
            config={**inputs,"diagnostic_contract_sha256":contract_sha,"split_sha256":contract["nested_registration"]["sha256"],
                    "entrypoint_sha256":contract["code_sha256"]["run_public_nested_v6.py"]})
        exit_code=1; arm_dir=output_dir/"runs"/arm; arm_dir.parent.mkdir(exist_ok=True)
        try:
            def progress(event):
                if event["phase"]=="inner":
                    tracker.log_inner(event["held_outer_source"],event["regularization"],event["metrics"],event["training_diagnostics"])
                else:
                    tracker.log_outer(event["held_outer_source"],event["metrics"],event["training_diagnostics"],
                        regularization=None if arm=="control" else event["regularization"])
                print(json.dumps({"arm":arm,"phase":event["phase"],"fold":event["fold_index"]+1,
                    "held_outer_source":event["held_outer_source"],"lambda":event["regularization"],
                    "primary_mse":event["metrics"]["target_pooled_mse"]}),flush=True)
            provenance={"diagnostic_contract_sha256":contract_sha,"role_manifest_sha256":contract["registration"]["sha256"],
                "nested_manifest_sha256":contract["nested_registration"]["sha256"],"inputs":inputs}
            summary=model.run_arm(arrays,roles,outer_tables,nested,inner_tables,arm,arm_dir,provenance,progress=progress)
            verify_summary(summary,arm,contract_sha,contract=contract,roles=roles,nested=nested)
            audit=model.verify_artifacts(arm_dir,summary,arrays,roles,outer_tables,nested,inner_tables)
            verify_audit(audit,arm)
            tracker.log_outer("all",summary["aggregate"]); exit_code=0
        finally: tracker.finish(exit_code)
        tp=output_dir/"tracking"/arm/"tracking.json"; receipt=authenticated_json(tp,sha256_file(tp))
        n=9 if arm=="control" else 33
        for key,v in {"execution_status":"completed","training_exit_code":0,"event_count":n,
            "tracking_errors":0,"mode":"offline","stage":"validation","cloud_synced":False}.items():
            equal(receipt.get(key),v,"tracking "+key)
        summaries[arm]=summary
        runs.append({"arm":arm,"summary_path":str((arm_dir/"summary.json").resolve()),
            "summary_sha256":sha256_file(arm_dir/"summary.json"),"tracking_path":str(tp.resolve()),
            "tracking_sha256":sha256_file(tp),"run_id":receipt["run_id"],"tracking_events":n,"tracking_errors":0,
            "cloud_synced":False,"artifact_audit":audit})
    if len({r["run_id"] for r in runs})!=7: raise ValueError("Distinct W&B runs required")
    validate_contract(contract_path,contract_sha); completed_at=datetime.now(timezone.utc).isoformat()
    suite={"schema":SCHEMA,"completed":True,"completed_at":completed_at,
        "diagnostic_contract_path":str(contract_path.resolve()),"diagnostic_contract_sha256":contract_sha,
        "decision":screen(summaries),"runs":runs,"source_metrics":{a:s["source_metrics"] for a,s in summaries.items()},
        "pretraining_performed":False,"posttraining_performed":False,"promoted":False,"submission_performed":False,
        "new_raw_expression_read":False,"gpu_used":False,"download_performed":False,"cloud_synced":False,
        "tracking_events":207,"tracking_errors":0,"runtime":{"hostname":socket.gethostname(),"device":"cpu"}}
    dump_new(output_dir/"suite.json",suite)
    complete={"schema":SCHEMA,"completed":True,"completed_at":completed_at,
        "suite_sha256":sha256_file(output_dir/"suite.json"),"diagnostic_contract_sha256":contract_sha,
        "screen_passed":suite["decision"]["passed"],"tracking_events":207,"tracking_errors":0,
        "cloud_synced":False,"promoted":False,"submission_performed":False}
    dump_new(output_dir/"complete.json",complete); print(json.dumps(complete,sort_keys=True),flush=True)
    return 0

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","run"))
    parser.add_argument("--role-manifest",type=Path); parser.add_argument("--role-manifest-sha256")
    parser.add_argument("--protocol",type=Path); parser.add_argument("--contract",type=Path)
    parser.add_argument("--contract-sha256"); parser.add_argument("--output-dir",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.action=="register":
        if args.role_manifest is None or not args.role_manifest_sha256 or args.protocol is None:
            parser.error("Registration requires role manifest, SHA and protocol")
        print(json.dumps({"contract_sha256":register(args.role_manifest,args.role_manifest_sha256,args.protocol,args.output_dir)}))
        return 0
    if args.contract is None or not args.contract_sha256: parser.error("Execution contract and SHA required")
    return run(args.contract,args.contract_sha256,args.output_dir)

if __name__=="__main__": raise SystemExit(main())
