"""Register/materialize source-fit-only GenePT transforms and diagnostic arms.

No expression, predictive model fitting, GPU, network, W&B upload or submission.
Source roles, quarantine, seeds and artifact identities precede feature fitting.
"""
from __future__ import annotations
import argparse
import io
import json
import os
from pathlib import Path
import resource
import signal
import time
import numpy as np
import convert_genept_numeric_v11 as numeric
import genept_conditioning_v11 as conditioning

SCHEMA = "genept-source-conditioning-materialization-v11"
ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"artifacts/public_flow/genept_numeric_v11_20260910"
CONVERSION_SHA = "f565eb23256722ed3b85d5960a71086138670aff55d58d2c2660dc8fe23cdc57"
FEATURE_SHA = "f1c29e064be43a6a2ebfc03371fb23537edf401582e8d5e2187967ebd3b83536"
QUARANTINE = sorted(["BRD1","BRPF1","LTB4R2","NOP9","MBD4","MED1","SKP2","UXT"])
SEEDS = (20260926,20260927)
require, dump, digest = numeric.require, numeric.fresh_json, numeric.digest


def policy():
    return {"seeds":list(SEEDS),"quarantine":QUARANTINE,"dimension":1536,
        "arms":["correct","masked","shuffled"],"mean_rms_scope":"available source-fit targets after quarantine",
        "global_held_targets_response_excluded":98,"challenge_expression":False,"predictive_model_training":False,
        "automatic_training_admission":False,"network":False,"gpu":False,"cpu_threads":2,
        "rss_limit":2<<30,"address_space_limit":3<<30,"wall_seconds":120,"output_limit":128<<20}


def plan(protocol):
    complete = numeric.bound_json(BASE/"numeric/complete.json",CONVERSION_SHA)
    require(complete["completed"] is True and complete["feature_sha256"] == FEATURE_SHA
            and complete["feature_bytes"] == 4083232 and complete["automatic_training_admission"] is False,
            "Pinned completed numeric conversion required")
    parent = numeric.bound_json(BASE/"registration/contract.json",complete["contract_sha256"])
    require(complete["contract_sha256"] == "965efbbc88ee7e6f2309af3d487260b247d9b2618331607c33d2eb3e59b36987",
            "Pinned conversion registration required")
    held = sorted({t for roles in parent["source_roles"].values() for t in roles["tune_targets"]})
    require(len(held) == 98 and len(parent["selected_targets"]) == 787,"Exact inherited roles required")
    require(sorted(t for g in complete["vector_sharing"]["binary64_content"]["duplicate_groups"] for t in g) == QUARANTINE,
            "Raw duplicate audit differs")
    protocol = Path(protocol).resolve()
    return {"schema":SCHEMA,"policy":policy(),"numpy_version":np.__version__,"numeric_completion_sha256":CONVERSION_SHA,
        "conversion_contract_sha256":complete["contract_sha256"],"numeric_feature_sha256":FEATURE_SHA,
        "numeric_feature_bytes":complete["feature_bytes"],"selected_targets":parent["selected_targets"],
        "sources":parent["source_roles"],"global_held_targets":held,"official_targets":parent["official_target_names"],
        "protocol":{"path":str(protocol),"sha256":digest(protocol)},
        "code_sha256":{name:digest(Path(__file__).with_name(name)) for name in
            ("materialize_genept_conditioning_v11.py","genept_conditioning_v11.py","convert_genept_numeric_v11.py")}}


def register(protocol,out):
    numeric.runtime_guard()
    require(not os.path.lexists(out),"Fresh registration required")
    record = {**plan(protocol),"registered_at":numeric.now()}
    out = Path(out).resolve()
    out.mkdir(mode=0o700)
    dump(out/"contract.json",record)
    return digest(out/"contract.json")


def write_npz(path, arrays):
    require(all(isinstance(a,np.ndarray) and a.dtype.kind in "biufUS" for a in arrays.values()),"Numeric/text NPZ only")
    require(all(a.dtype.kind != "f" or np.isfinite(a).all() for a in arrays.values()),"Finite numeric arrays required")
    require(sum(a.nbytes for a in arrays.values()) < 64<<20,"Per-artifact size bound")
    with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600),"wb") as stream:
        np.savez_compressed(stream,**arrays)
        stream.flush(); os.fsync(stream.fileno())
    with np.load(path,allow_pickle=False) as replay:
        require(set(replay.files)==set(arrays) and all(np.array_equal(v,replay[k]) and v.dtype == replay[k].dtype
                for k,v in arrays.items()),"Numeric artifact replay differs")
    return {"path":str(path),"sha256":digest(path),"bytes":path.stat().st_size}


def execute(contract,sha,out):
    numeric.runtime_guard()
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(3<<30,3<<30))
    previous=signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError("Feature materialization wall cap")))
    signal.alarm(120)
    try:
        record=numeric.bound_json(contract,sha)
        expected=plan(record["protocol"]["path"])
        require({k:v for k,v in record.items() if k!="registered_at"}==expected,"Feature contract reconstruction failed")
        require(not os.path.lexists(out),"Fresh materialization output required")
        out=Path(out).resolve(); out.mkdir(mode=0o700)
        dump(out/"started.json",{"schema":SCHEMA,"contract_sha256":sha,"started_at":numeric.now()})
        payload=numeric.read_regular(BASE/"numeric/features.npz",8<<20,payload=True)
        import hashlib
        require(len(payload)==record["numeric_feature_bytes"] and hashlib.sha256(payload).hexdigest()==FEATURE_SHA,
                "Numeric source authentication failed")
        with np.load(io.BytesIO(payload),allow_pickle=False) as source:
            require(set(source.files)=={"target_ids","embeddings","present","source_vector_object_id","source_vector_sha256"},
                    "Exact numeric source fields required")
            table=conditioning.prepare_features(source["target_ids"].tolist(),source["embeddings"],source["present"],
                source_vector_object_id=source["source_vector_object_id"])
        require(list(table.target_ids)==record["selected_targets"],"Source target order differs")
        filtered,quarantine=conditioning.quarantine_duplicate_features(table,raw_artifact_sha256=FEATURE_SHA)
        require(quarantine["quarantined_target_names"]==QUARANTINE and int(filtered.present.sum())==774,
                "Registered exact duplicate quarantine differs")
        named_roles={s+"_"+r:ts for s,roles in record["sources"].items() for r,ts in roles.items()}
        named_roles["official"]=record["official_targets"]
        sharing=conditioning.audit_sharing(filtered,named_roles)
        require(sharing["requires_training_admission_review"] is False,"Unresolved filtered cross-role sharing")
        dump(out/"quarantine.json",quarantine); dump(out/"filtered_sharing.json",sharing)
        index={t:i for i,t in enumerate(filtered.target_ids)}
        coverage={s:{r:{"total":len(ts),"covered":sum(int(filtered.present[index[t]]) for t in ts)}
            for r,ts in roles.items()} for s,roles in record["sources"].items()}
        coverage["official"]={"total":300,"covered":sum(int(filtered.present[index[t]]) for t in record["official_targets"])}
        require(coverage=={"replogle_k562":{"fit_targets":{"total":299,"covered":292},"tune_targets":{"total":75,"covered":75}},
            "nadig_jurkat":{"fit_targets":{"total":90,"covered":87},"tune_targets":{"total":23,"covered":23}},
            "official":{"total":300,"covered":297}},"Exact source/held/official coverage differs")
        outputs={}
        for source,roles in record["sources"].items():
            directory=out/source; directory.mkdir(mode=0o700)
            stats=conditioning.fit_transform(filtered,source_id=source,fit_targets=roles["fit_targets"],
                held_targets=record["global_held_targets"],official_targets=record["official_targets"])
            serialized=stats.to_dict()
            dump(directory/"transform.json",serialized)
            restored=conditioning.GenePTTransform.from_dict(numeric.bound_json(directory/"transform.json",digest(directory/"transform.json")))
            require(restored.to_dict()==serialized,"Transform round-trip differs")
            full=conditioning.transform_features(filtered,restored)
            fit_rows=[index[t] for t in stats.available_fit_targets]
            fit_values=full.embeddings[fit_rows].astype(np.float64)
            require(np.max(np.abs(fit_values.mean(0)))<1e-6 and abs(float(np.mean(fit_values**2))-1)<1e-6,
                    "Fitting-only mean/RMS replay failed")
            full_binding=write_npz(directory/"all_targets.npz",{"target_ids":np.asarray(full.target_ids),
                "embeddings":full.embeddings,"present":full.present})
            local=conditioning.transform_features(filtered,restored,target_ids=sorted(roles["fit_targets"]+roles["tune_targets"]))
            seed_outputs={}
            for seed in SEEDS:
                arms=conditioning.build_arms(local,fit_targets=roles["fit_targets"],held_targets=roles["tune_targets"],seed=seed)
                require(all(np.array_equal(a.present,local.present) for a in arms.values()),"Arm availability differs")
                arrays={"target_ids":np.asarray(local.target_ids),"present":local.present,
                    **{name:arm.embeddings for name,arm in arms.items()},
                    "shuffled_donor_target_ids":np.asarray(arms["shuffled"].donor_target_ids)}
                seed_outputs[str(seed)]=write_npz(directory/f"arms_{seed}.npz",arrays)
                report_path=directory/f"arms_{seed}.json"
                dump(report_path,{name:arm.report for name,arm in arms.items()})
                seed_outputs[str(seed)]["report"]={"path":str(report_path),"sha256":digest(report_path)}
            outputs[source]={"transform_sha256":digest(directory/"transform.json"),"available_fit_targets":len(stats.available_fit_targets),
                "scalar_rms":stats.scalar_rms,"all_targets":full_binding,"arms":seed_outputs}
        require(expected==plan(record["protocol"]["path"]),"Feature provenance changed during materialization")
        require(digest(BASE/"numeric/features.npz")==FEATURE_SHA,"Numeric source changed during materialization")
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
        size=sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        require(peak<=2<<30 and size+(1<<20)<=128<<20 and time.monotonic()-started<120,"Final feature resource bound")
        result={"schema":SCHEMA,"completed":True,"completed_at":numeric.now(),"contract_sha256":sha,
            "raw_feature_sha256":FEATURE_SHA,"quarantine_sha256":digest(out/"quarantine.json"),
            "filtered_sharing_sha256":digest(out/"filtered_sharing.json"),"coverage":coverage,"sources":outputs,
            "resources":{"peak_rss_bytes":peak,"elapsed_seconds":time.monotonic()-started,"artifact_bytes_before_receipt":size},
            "predictive_model_training_performed":False,"embedding_statistics_fitted":True,
            "challenge_expression_read":False,"responses_read":False,"gpu_used":False,"network_used":False,
            "automatic_training_admission":False,"all_numpy_artifacts_replayed":True}
        dump(out/"complete.json",result)
        return result
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest="action",required=True)
    r=sub.add_parser("register")
    for name in ("protocol","out"): r.add_argument("--"+name,required=True)
    r=sub.add_parser("run")
    for name in ("contract","sha","out"): r.add_argument("--"+name,required=True)
    args=vars(parser.parse_args()); action=args.pop("action")
    print(json.dumps(register(**args) if action=="register" else execute(**args),sort_keys=True),flush=True)
