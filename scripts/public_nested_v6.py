"""Public target-by-context experiment with the frozen v4 sparse output.

Only authenticated existing public cache arrays are admitted by the caller.
This is a small diagnostic, not a flow parent, count emitter or submission.
"""
from __future__ import annotations
from dataclasses import replace
import json
import os
from pathlib import Path
import socket
import numpy as np

import public_joint_target_source_v3 as v3
import public_sparse_hurdle_v4 as sparse
import public_nested_metrics_v6 as metrics
import public_target_context_kernel_v6 as kernel
from public_mean_effect_baseline import balanced_weights, save_arrays
from train_public_flow_pilot import load_npz, write_new
from wandb_training import finite_scalar

ARMS = ("control", "context", "additive", "true", "shuffled_20260921", "shuffled_20260922", "shuffled_20260923")
SHUFFLED_ARMS = ARMS[4:]
COMMON_KEYS = ("fit_targets", "held_targets", "fit_source", "held_source",
    "aligned_target_ids", "aligned_values", "aligned_present", "feature_ids",
    "feature_signature_json", "feature_receipt_json", "positive_prior", "prior_positive_counts",
    "training_positive_support", "prior_source_ids", "prior_original_rows", "prior_pool_kind",
    "training_group_ids", "training_response")


GRID = (0.1, 1.0, 10.0)
PREDICTION_KEYS = (*sparse.PREDICTION_KEYS, "control_parent_cache_rows", "treated_parent_cache_rows")

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)

def policy():
    return {"schema":"public-nested-regularization-model-v6","arms":list(ARMS),
        "kernel":kernel.policy(),"decoder":sparse.policy(),"metrics":metrics.policy(),
        "regularization_grid":list(GRID),"selection_metric":"inner target_pooled_mse",
        "selection_rule":"minimum primary MSE; exact ties choose strongest regularization",
        "selection_scope":"inner same-source held targets with globally disjoint reference rows",
        "hyperparameter_selection":"learned arms only; control has no selection",
        "outer_response_selection":False,"inner_reference_cells":16,"inner_tuning_anchor_cells":16,
        "outer_reference_cells":32,
        "inner_context_donors":"independent context pool allowed as frozen inference donor, never prior/response fitting",
        "outer_evaluation":"selected lambda per arm/direction; refit42 on original permitted pools",
        "posttraining":False,"count_emitter":False,"automatic_promotion":False,
        "prior_public_development_exposure_acknowledged":True}


def feature_arm(arm):
    if arm not in ARMS:
        raise ValueError("Unknown interaction arm")
    return "true" if arm == "additive" else arm


def fit_arguments(arrays, fold):
    return dict(training_targets=tuple(str(arrays["targets"][i]) for i in fold.fit_indices),
        training_sources=tuple(str(arrays["sources"][i]) for i in fold.fit_indices),
        context=arrays["context"][list(fold.fit_indices)])


def training_diagnostics(decoder, response, arrays, fold, aligned):
    result = dict(kernel.training_diagnostics(decoder, response, regularization=decoder.regularization))
    targets = tuple(str(arrays["targets"][i]) for i in fold.fit_indices)
    weights = balanced_weights(targets, tuple(str(arrays["sources"][i]) for i in fold.fit_indices))
    heads = []
    for group in fold.fit_indices:
        packet = v3._subset(aligned, (str(arrays["targets"][group]),))
        heads.append(decoder.predict_delta(packet, arrays["context"][group:group+1],
            is_ntc=np.array([False]))[0])
    heads = np.asarray(heads)
    width = arrays["control"].shape[1]
    unchanged = []
    for head, group in zip(heads, fold.fit_indices):
        reference = arrays["control"][arrays["control_group"] == group]
        n = len(reference)
        count = np.sum(reference > 0, axis=0)
        q = count / n
        smoothed = (32*q+.5)/33
        delta = np.clip(head[:width], -sparse.OCCUPANCY_CLIP, sparse.OCCUPANCY_CLIP)
        shifted = 1 / (1 + np.exp(-(np.log(smoothed)-np.log1p(-smoothed)+delta)))
        requested = np.clip(q+shifted-smoothed, 0, 1)
        requested[delta == 0] = q[delta == 0]
        desired = np.floor(n*requested+.5).astype(np.int64)
        unchanged.append(float(np.mean(desired == count)))
    result["training_unchanged_count_fraction"] = float(weights @ unchanged)
    if any(type(x) not in (int, float) or not np.isfinite(x) or x < 0 for x in result.values()):
        raise ValueError("Invalid training-only diagnostic")
    return result


def model_arrays(decoder, prior, response, fold, aligned, receipt, table):
    result = {"fit_targets": np.asarray(fold.fit_targets), "held_targets": np.asarray(fold.held_targets),
        "fit_source": np.asarray(fold.fit_source), "held_source": np.asarray(fold.held_source),
        "aligned_target_ids": np.asarray(aligned.target_ids), "aligned_values": aligned.values,
        "aligned_present": aligned.present, "feature_ids": np.asarray(table.feature_ids),
        "feature_signature_json": np.asarray(json.dumps(aligned.signature)),
        "feature_receipt_json": np.asarray(json.dumps(receipt, sort_keys=True)),
        "positive_prior": prior.values, "prior_positive_counts": prior.positive_counts,
        "training_positive_support": prior.supported, "prior_source_ids": prior.source_ids,
        "prior_original_rows": prior.original_rows, "prior_pool_kind": prior.pool_kind,
        "training_group_ids": np.asarray(fold.fit_indices, dtype=np.int64), "training_response": response}
    result.update({"kernel_"+k: value for k, value in kernel.decoder_arrays(decoder).items()})
    return result


def finish_report(report, number, fold, receipt, decoder, prior, train_report):
    report.update(array_prefix=f"fold_{number}_", feature_receipt=receipt,
        target_transform=decoder.target_transform.provenance() if decoder.target_transform else None,
        prior_unique_training_rows=len(prior.original_rows),
        prior_unique_anchor_rows=int(np.sum(prior.pool_kind == 1)),
        prior_unique_treated_rows=int(np.sum(prior.pool_kind == 2)),
        unsupported_genes=int(np.sum(~prior.supported)), training_diagnostics=train_report)



def inner_fold(arrays, outer, record):
    info=record["inner_folds"][outer.target_fold]
    fit,tune=tuple(info["fit_targets"]),tuple(info["tuning_targets"])
    if (not fit or not tune or len(set(fit))!=len(fit) or len(set(tune))!=len(tune)
        or set(fit)&set(tune) or set(fit)|set(tune)!=set(outer.fit_targets)):
        raise ValueError("Inner target roles must partition outer fitting targets")
    targets,sources=arrays["targets"].astype(str),arrays["sources"].astype(str)
    fi=tuple(i for i in outer.fit_indices if targets[i] in fit)
    ei=tuple(i for i in outer.fit_indices if targets[i] in tune)
    hi=tuple(i for i,t in enumerate(targets) if t in tune)
    oi=tuple(i for i in range(len(targets)) if i not in set(fi)|set(hi))
    if ({str(targets[i]) for i in fi}!=set(fit) or {str(targets[i]) for i in ei}!=set(tune)
        or any(sources[i]!=outer.fit_source for i in fi+ei)):
        raise ValueError("Inner source/target groups unauthorized")
    return kernel.InnerTargetFold(fold_id=outer.fold_id+"__inner_v6",target_fold=outer.target_fold+"__inner_v6",
        fit_source=outer.fit_source,held_source=outer.fit_source,fit_targets=fit,held_targets=tune,
        fit_indices=fi,evaluation_indices=ei,held_target_indices_all_sources=hi,other_excluded_indices=oi,
        outer_fold_id=outer.fold_id)

def inner_view(arrays, fold, record):
    allowed=set(fold.fit_indices)|set(fold.evaluation_indices)
    pools={(p["source"],p["batch"]):p for p in record["anchor_pools"]}
    if len(pools)!=len(record["anchor_pools"]): raise ValueError("Duplicate anchor pool")
    chosen=[]; identities=[set(),set()]
    for group in sorted(allowed):
        source,batch=str(arrays["sources"][group]),str(arrays["batches"][group])
        p=pools[(source,batch)]; fr,tr=p["fit_reference_rows"],p["tuning_anchor_rows"]
        if (len(fr)!=16 or len(tr)!=16 or any(type(x) is not int or x<0 for x in fr+tr)
            or len(set(fr+tr))!=32): raise ValueError("Globally disjoint16/16 references required")
        ci=np.flatnonzero(arrays["control_group"]==group)
        if len(ci)!=32 or set(map(int,arrays["control_rows"][ci]))!=set(fr+tr):
            raise ValueError("Anchor original rows differ from registration")
        role=0 if group in fold.fit_indices else 1
        wanted=set(fr if role==0 else tr)
        selected=[int(i) for i in ci if int(arrays["control_rows"][i]) in wanted]
        chosen.extend(selected)
        identities[role].update((source,int(arrays["control_rows"][i])) for i in selected)
    if identities[0]&identities[1]: raise ValueError("Reference rows cross fitting/tuning roles")
    view={k:arrays[k] for k in ("targets","sources","batches","genes")}
    ci=np.asarray(chosen,dtype=np.int64)
    ti=np.flatnonzero(np.isin(arrays["group"],sorted(allowed)))
    xi=np.flatnonzero(np.isin(arrays["context_control_group"],sorted(allowed)))
    for key in ("control","control_group","control_rows"): view[key]=arrays[key][ci]
    for key in ("treated","group","treated_rows"): view[key]=arrays[key][ti]
    for key in ("context_control","context_control_group","context_control_rows"): view[key]=arrays[key][xi]
    view["context"]=np.zeros_like(arrays["context"])
    view["context"][sorted(allowed)]=arrays["context"][sorted(allowed)]
    view["_control_parent_cache_rows"],view["_treated_parent_cache_rows"]=ci,ti
    for value in view.values(): value.setflags(write=False)
    return view

def positive_prior(arrays, fold):
    if type(fold) is v3.JointFold: return sparse.training_positive_prior(arrays,fold)
    if type(fold) is not kernel.InnerTargetFold or fold.fit_source!=fold.held_source:
        raise ValueError("Explicit same-source inner role required")
    ts,ss=arrays["targets"].astype(str),arrays["sources"].astype(str)
    if (set(fold.fit_targets)&set(fold.held_targets)
        or set(fold.fit_indices)&set(fold.held_target_indices_all_sources)
        or set(fold.fit_indices)&set(fold.evaluation_indices)
        or any(i<0 or i>=len(ts) or ss[i]!=fold.fit_source or ts[i] not in fold.fit_targets for i in fold.fit_indices)):
        raise ValueError("Unauthorized inner prior fitting role")
    unique={}
    for field,label,rowkey,kind in (("control","control_group","control_rows",1),("treated","group","treated_rows",2)):
        for i in np.flatnonzero(np.isin(arrays[label],fold.fit_indices)):
            group=int(arrays[label][i]); original=arrays[rowkey][i]
            if not isinstance(original,(int,np.integer)) or original<0: raise ValueError("Invalid original row")
            key=str(ss[group]),int(original); value=np.asarray(arrays[field][i],dtype=np.float64)
            if value.ndim!=1 or not np.isfinite(value).all() or np.any(value<0): raise ValueError("Invalid inner prior expression")
            if key in unique:
                kind0,value0=unique[key]
                if kind0!=kind or not np.array_equal(value0,value): raise ValueError("Conflicting repeated original row")
            else: unique[key]=kind,value.copy()
    if not unique: raise ValueError("No fitting cells for inner prior")
    keys=sorted(unique); matrix=np.asarray([unique[k][1] for k in keys])
    counts=(matrix>0).sum(0).astype(np.int64)
    maxima=matrix.max(0); scales=np.where(maxima>0,maxima,1)
    values=scales*(np.sum(matrix/scales,axis=0)/np.maximum(counts,1))
    if not np.isfinite(values).all(): raise ValueError("Inner prior arithmetic overflow")
    result=sparse.PositivePrior(values,counts,np.asarray([k[0] for k in keys]),
        np.asarray([k[1] for k in keys],dtype=np.int64),np.asarray([unique[k][0] for k in keys],dtype=np.int8))
    for value in (result.values,result.positive_counts,result.source_ids,result.original_rows,result.pool_kind):
        value.setflags(write=False)
    return result

def select_regularization(candidates):
    if not isinstance(candidates,list) or len(candidates)!=3 or [c.get("regularization") for c in candidates]!=list(GRID):
        raise ValueError("Exactly the ordered lambda grid required")
    for c in candidates:
        lam,score=c["regularization"],c["metrics"]["target_pooled_mse"]
        if type(lam) is not float or not finite_scalar(score) or score<0:
            raise ValueError("Invalid inner lambda/score")
    return min(candidates,key=lambda c:(c["metrics"]["target_pooled_mse"],-c["regularization"]))["regularization"]

def evaluate(arrays, fold, aligned, decoder, prior):
    report,output=metrics.evaluate_fold(arrays,fold,aligned,decoder,prior)
    for name in ("control","treated"):
        indices=output[name+"_cache_rows"]; mapping=arrays.get("_"+name+"_parent_cache_rows")
        output[name+"_parent_cache_rows"]=indices.copy() if mapping is None else mapping[indices]
    return report,output

def augment(report, decoder, prior, response, arrays, fold, aligned, receipt, prefix, lam):
    report.update(array_prefix=prefix,feature_receipt=receipt,
        target_transform=decoder.target_transform.provenance() if decoder.target_transform else None,
        prior_unique_training_rows=len(prior.original_rows),prior_unique_anchor_rows=int(np.sum(prior.pool_kind==1)),
        prior_unique_treated_rows=int(np.sum(prior.pool_kind==2)),unsupported_genes=int(np.sum(~prior.supported)),
        training_diagnostics=training_diagnostics(decoder,response,arrays,fold,aligned),
        regularization=lam,validation_kind=decoder.validation_kind,outer_fold_id=decoder.outer_fold_id)

def one_fit(arrays, fold, table, arm, lam, prefix):
    aligned,receipt=v3.role_features(table,fold.fit_targets,fold.held_targets,arm=feature_arm(arm),fold_id=fold.target_fold)
    receipt={**receipt,"consumer_arm":arm}
    prior=positive_prior(arrays,fold); response=sparse.training_responses(arrays,fold,prior); args=fit_arguments(arrays,fold)
    decoder=kernel.fit_kernel_decoder(args["training_targets"],args["training_sources"],args["context"],response,
        aligned=aligned,arm=arm,fold=fold,representation=sparse.REPRESENTATION,regularization=lam)
    report,output=evaluate(arrays,fold,aligned,decoder,prior)
    augment(report,decoder,prior,response,arrays,fold,aligned,receipt,prefix,lam)
    return report,output,model_arrays(decoder,prior,response,fold,aligned,receipt,table)

def selection_record(arm,candidates,selected):
    return {"performed":arm!="control","scope":"inner_only_disjoint_reference_rows",
        "outer_response_selection":False,"selected_regularization":selected,"candidates":candidates,
        "tie_rule":"exact primary tie chooses stronger regularization"}

def run_arm(arrays, registration, outer_tables, nested_record, inner_tables, arm, output_dir, provenance, progress=None):
    if arm not in ARMS: raise ValueError("Unknown nested arm")
    if not isinstance(provenance.get("diagnostic_contract_sha256"),str) or len(provenance["diagnostic_contract_sha256"])!=64:
        raise ValueError("Frozen execution provenance required")
    folds=v3.joint_folds(tuple(arrays["targets"].astype(str)),tuple(arrays["sources"].astype(str)),registration)
    if set(outer_tables)!={f.target_fold for f in folds} or set(inner_tables)!=set(outer_tables):
        raise ValueError("Exact outer and inner vocabularies required")
    output_dir=Path(output_dir)
    if os.path.lexists(output_dir): raise FileExistsError("Fresh nested arm output required")
    output_dir.mkdir(mode=0o700)
    reports,weights,predictions,events=[],{},{},[]
    def save(prefix,output,state):
        weights.update({prefix+k:v for k,v in state.items()})
        predictions.update({prefix+k:v for k,v in output.items()})
    def emit(phase,number,outer,lam,report):
        event={"phase":phase,"fold_index":number,"held_outer_source":outer.held_source,"regularization":lam,
            "metrics":report["metrics"],"training_diagnostics":report["training_diagnostics"]}
        events.append(event)
        if progress: progress(event)
    for number,outer in enumerate(folds):
        candidates=[]
        if arm!="control":
            inner=inner_fold(arrays,outer,nested_record); view=inner_view(arrays,inner,nested_record)
            for index,lam in enumerate(GRID):
                prefix=f"fold_{number}_inner_{index}_"
                report,output,state=one_fit(view,inner,inner_tables[outer.target_fold],arm,lam,prefix)
                save(prefix,output,state); candidates.append(report); emit("inner",number,outer,lam,report)
            selected=select_regularization(candidates)
        else: selected=10.0
        prefix=f"fold_{number}_outer_"
        report,output,state=one_fit(arrays,outer,outer_tables[outer.target_fold],arm,selected,prefix)
        save(prefix,output,state)
        report["inner_selection"]=selection_record(arm,candidates,selected)
        reports.append(report); emit("outer",number,outer,selected,report)
    sources,aggregate=sparse._final_metrics(reports)
    weights["genes"]=predictions["genes"]=arrays["genes"]
    summary={"schema":"public-nested-regularization-summary-v6","status":"completed_unpromoted_diagnostic",
        "stage":"validation","arm":arm,"policy":policy(),"folds":reports,"source_metrics":sources,"aggregate":aggregate,
        "provenance":provenance,"diagnostic_contract_sha256":provenance["diagnostic_contract_sha256"],
        "weights_sha256":save_arrays(output_dir/"weights.npz",weights),
        "predictions_sha256":save_arrays(output_dir/"predictions.npz",predictions),
        "exact_ntc_identity_verified":all(r["exact_ntc_identity_verified"] for r in reports),
        "exact_zero_effect_identity_verified":all(r["exact_zero_effect_identity_verified"] for r in reports),
        "hyperparameter_selection":arm!="control","inner_selection_performed":arm!="control",
        "selection_scope":"inner_only_disjoint_reference_rows","outer_response_selection":False,
        "count_emitter":False,"posttraining_performed":False,"promoted":False,"submission_performed":False,
        "H1_treated_read":False,"RPE1_treated_read":False,"challenge_treated_used":False,
        "outer_evaluation_roles_used_for_fitting":False,"prior_public_development_exposure_acknowledged":True,
        "runtime":{"hostname":socket.gethostname(),"device":"cpu","numpy_version":np.__version__}}
    write_new(output_dir/"steps.jsonl",b"".join((canonical(e)+"\n").encode() for e in events))
    write_new(output_dir/"summary.json",(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+"\n").encode())
    return summary

def replay_entry(arrays,fold,table,arm,lam,prefix,saved,weights,predictions):
    aligned,receipt=v3.role_features(table,fold.fit_targets,fold.held_targets,arm=feature_arm(arm),fold_id=fold.target_fold)
    receipt={**receipt,"consumer_arm":arm}
    prior=positive_prior(arrays,fold); response=sparse.training_responses(arrays,fold,prior)
    decoder=kernel.restore_kernel_decoder({k:weights[prefix+"kernel_"+k] for k in kernel.decoder_keys(arm)},
        arm=arm,fold=fold,aligned=aligned,response=response,regularization=lam,**fit_arguments(arrays,fold))
    for key,value in model_arrays(decoder,prior,response,fold,aligned,receipt,table).items():
        if weights[prefix+key].dtype!=np.asarray(value).dtype or not np.array_equal(weights[prefix+key],value):
            raise ValueError("Nested training-state mismatch: "+key)
    report,output=evaluate(arrays,fold,aligned,decoder,prior)
    for key,value in output.items():
        if predictions[prefix+key].dtype!=np.asarray(value).dtype or not np.array_equal(predictions[prefix+key],value):
            raise ValueError("Nested prediction mismatch: "+key)
    augment(report,decoder,prior,response,arrays,fold,aligned,receipt,prefix,lam)
    report=json.loads(json.dumps(report,allow_nan=False))
    sparse._same_tree(saved,report,"nested replay")
    return report

def verify_artifacts(output_dir,summary,arrays,registration,outer_tables,nested_record,inner_tables):
    summary=json.loads(json.dumps(summary,allow_nan=False))
    if summary.get("schema")!="public-nested-regularization-summary-v6" or summary.get("arm") not in ARMS:
        raise ValueError("Invalid nested summary")
    arm=summary["arm"]
    for key in ("count_emitter","posttraining_performed","promoted","submission_performed",
                "H1_treated_read","RPE1_treated_read","challenge_treated_used","outer_evaluation_roles_used_for_fitting"):
        if summary.get(key) is not False: raise ValueError("Forbidden nested activity")
    for key in ("exact_ntc_identity_verified","exact_zero_effect_identity_verified","prior_public_development_exposure_acknowledged"):
        if summary.get(key) is not True: raise ValueError("Missing nested safety/exposure flag")
    if canonical(summary["policy"])!=canonical(policy()): raise ValueError("Nested policy changed")
    if (summary.get("hyperparameter_selection") is not (arm!="control")
        or summary.get("inner_selection_performed") is not (arm!="control")
        or summary.get("outer_response_selection") is not False): raise ValueError("Nested selection scope changed")
    folds=v3.joint_folds(tuple(arrays["targets"].astype(str)),tuple(arrays["sources"].astype(str)),registration)
    if len(summary["folds"])!=len(folds): raise ValueError("Outer fold roster changed")
    prefixes=[f"fold_{i}_outer_" for i in range(len(folds))]
    if arm!="control": prefixes += [f"fold_{i}_inner_{j}_" for i in range(len(folds)) for j in range(3)]
    state_keys=COMMON_KEYS+tuple("kernel_"+k for k in kernel.decoder_keys(arm))
    weights=load_npz(Path(output_dir)/"weights.npz",summary["weights_sha256"],{"genes"}|{p+k for p in prefixes for k in state_keys})
    predictions=load_npz(Path(output_dir)/"predictions.npz",summary["predictions_sha256"],{"genes"}|{p+k for p in prefixes for k in PREDICTION_KEYS})
    if not np.array_equal(weights["genes"],arrays["genes"]) or not np.array_equal(predictions["genes"],arrays["genes"]):
        raise ValueError("Nested gene axis mismatch")
    reports=[]
    for number,(outer,saved) in enumerate(zip(folds,summary["folds"])):
        candidates=[]
        if arm!="control":
            inner=inner_fold(arrays,outer,nested_record); view=inner_view(arrays,inner,nested_record)
            original=saved["inner_selection"]["candidates"]
            if len(original)!=3: raise ValueError("Nested candidate roster mismatch")
            for index,lam in enumerate(GRID):
                candidates.append(replay_entry(view,inner,inner_tables[outer.target_fold],arm,lam,
                    f"fold_{number}_inner_{index}_",original[index],weights,predictions))
            selected=select_regularization(candidates)
        else: selected=10.0
        selection=selection_record(arm,candidates,selected)
        sparse._same_tree(saved["inner_selection"],selection,"inner selection")
        plain={k:v for k,v in saved.items() if k!="inner_selection"}
        report=replay_entry(arrays,outer,outer_tables[outer.target_fold],arm,selected,
            f"fold_{number}_outer_",plain,weights,predictions)
        report["inner_selection"]=selection; reports.append(report)
    sources,aggregate=sparse._final_metrics(reports)
    sparse._same_tree(summary["source_metrics"],sources,"source metrics")
    sparse._same_tree(summary["aggregate"],aggregate,"aggregate")
    return {"passed":True,"ridge_refits":0,"saved_head_replay":True,"training_prior_replay":True,
        "kernel_system_verified":True,"training_only_transforms_verified":True,"deterministic_donor_replay":True,
        "original_row_mapping_verified":True,"pooled_and_batch_metrics_replayed":True,"training_diagnostics_replayed":True,
        "inner_selection_replayed":True,"disjoint_inner_references_verified":True,
        "inner_candidates_replayed":0 if arm=="control" else 3*len(folds),
        "out_of_fold_groups":sum(len(f.evaluation_indices) for f in folds),
        "out_of_fold_source_targets":sum(len(f.held_targets) for f in folds)}
