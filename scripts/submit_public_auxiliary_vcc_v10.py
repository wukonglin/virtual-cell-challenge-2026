"""Submit one authenticated exploratory package; resume/poll the SAME entry.

Uses the installed official VCC client with its limit/in-flight checks enabled.
Token is read through a hidden prompt and never stored; private client state
contains expiring upload-resume metadata, not the API token.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import getpass
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import time

import public_training_readiness as safe

SCHEMA = "public-auxiliary-vcc-upload-v10"
ENDPOINT = "https://virtualcellchallenge.org"
MODEL = "GO_context_flow_v10_exploratory"
DESCRIPTION = ("Exploratory public-only CRISPRi flow: equal-weight K562/Jurkat true-GO models, "
    "fit-only GO vocabularies and released-control mean/std context. 482 modeled genes; "
    "18,051 others preserve native released-control counts. Original 7,097-gene control-depth "
    "inverse normalization and seeded unbiased integer rounding. No challenge-treated fitting, "
    "GenePT, PromoterAI, or Feng post-training. Public source-local validation did not beat "
    "unchanged controls; this submission measures challenge transfer, not a claimed improvement.")
require = safe.require


def dump(path, value):
    payload = (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
    require(len(payload) <= 16 << 20, "Upload receipt too large")
    flags = os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_NOFOLLOW",0)
    with os.fdopen(os.open(path,flags,0o600),"wb") as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())


def checked_package(path, sha):
    record = safe.read_json(path,sha)
    require(record.get("completed") is True and record.get("official_cli_version") == "0.2.0"
        and record.get("official_container_validated") is True,
        "Officially validated completed package required")
    prep = record.get("prep_result",{})
    require(prep.get("n_cells") == 360000 and prep.get("n_genes") == 18533
        and prep.get("normalization") == "counts-preserved" and prep.get("verified_targets") is True
        and prep.get("dry_run") is False and prep.get("cells_per_context") == {"A":120000,"B":120000,"C":120000}
        and type(prep.get("nnz")) is int and 0 < prep["nnz"] <= 4750000000,
        "Exact official count/axis/quota validation required")
    safe.binding(record.get("package"),json_document=False)
    generation = safe.binding(record.get("generation_completion"))
    require(generation.get("completed") is True and generation.get("submission_performed") is False,
        "Fresh completed generation required")
    return record


def public_status(entry):
    from vcc._scores import SCORE_KEYS
    keys = {"entry_id","status","error_code","model_name",*SCORE_KEYS}
    return {key:entry[key] for key in keys if key in entry}


def monitor(token, entry_id, out, seconds):
    from vcc import api, submit
    require(type(seconds) is int and 0 <= seconds <= 3600, "Bounded one-hour monitor required")
    deadline = time.monotonic()+seconds
    number = 1
    while (out/f"status_{number:05d}.json").exists(): number += 1
    previous = None
    while True:
        entry = public_status(api.get_submission(ENDPOINT,token,entry_id))
        entry.setdefault("entry_id",entry_id)
        require(entry["entry_id"] == entry_id, "Server returned a different entry")
        dump(out/f"status_{number:05d}.json", {"checked_at":datetime.now(timezone.utc).isoformat(),**entry})
        number += 1
        if entry != previous:
            print(json.dumps(entry,sort_keys=True),flush=True)
            previous = entry
        if submit.is_done(entry.get("status")) or time.monotonic() >= deadline:
            return entry
        time.sleep(min(30,max(0,deadline-time.monotonic())))


def execute(action, package_path, package_sha, out, seconds):
    require(action in {"submit","resume","status"} and type(seconds) is int and 0 <= seconds <= 3600,
        "Known action and bounded monitor required before any upload")
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3","cbsuvlaminck6"},
        "Run on an approved compute host, never head/login")
    require(importlib.metadata.version("vcc-cli") == "0.2.0", "Reviewed official VCC client0.2.0 required")
    package_path = Path(package_path).resolve()
    record = checked_package(package_path,package_sha)
    out = Path(out).resolve()
    if action == "submit":
        require(not os.path.lexists(out), "New entry requires a fresh upload directory; resume existing attempts instead")
        out.mkdir(mode=0o700)
        dump(out/"started.json", {"schema":SCHEMA,"created_at":datetime.now(timezone.utc).isoformat(),
            "package_completion":{"path":str(package_path),"sha256":package_sha},
            "model_name":MODEL,"script_sha256":safe.digest(__file__),"daily_limit_check":True,
            "one_inflight_check":True,"inflight_scope":"server-enforced per team; local lock per attempt",
            "token_stored":False})
    else:
        started = safe.read_json(out/"started.json")
        require(started["package_completion"] == {"path":str(package_path),"sha256":package_sha},
            "Resume/status must use the same authenticated package")
    # This override affects only expiring official-client resume state/locks.
    private = out/"private_vcc_state"
    if not private.exists(): private.mkdir(mode=0o700)
    require(not private.is_symlink(), "Private resume directory must not be a symlink")
    os.environ["VCC_CONFIG_DIR"] = str(private)
    from vcc import api, auth, submit
    from vcc.lock import SubmitLock
    token = getpass.getpass("VCC token (hidden; not stored): ")
    require(token.startswith("vcc_pat_") and "\n" not in token, "Expected VCC personal access token")
    try:
        if action == "status":
            entry_id = safe.read_json(out/"entry.json")["entry_id"]
            return monitor(token,entry_id,out,seconds)
        lock = SubmitLock(private/"locks/submit-default.lock")
        lock.acquire()
        try:
            resume = None
            if action == "resume":
                entry_id = safe.read_json(out/"entry.json")["entry_id"]
                current = api.get_submission(ENDPOINT,token,entry_id)
                require(current.get("entry_id",entry_id) == entry_id, "Resume server entry differs")
                if current.get("status") in {"launching","scoring","published","failed","hidden_admin","superseded"}:
                    return monitor(token,entry_id,out,seconds)
                require(current.get("status") in {"uploading","pending"}, "Unknown resume status; inspect existing entry")
                resume = auth.get_pending_upload("default",entry_id)
                require(resume is not None, "No upload to resume; poll the existing entry instead")
                require(resume.get("entry_id") == entry_id and resume.get("model_name") == MODEL
                    and isinstance(resume.get("local_path"),str)
                    and Path(resume["local_path"]).resolve() == Path(record["package"]["path"]).resolve(),
                    "Resume must use this same entry, model name and authenticated package")
            last_progress = 0.
            def event(kind, **data):
                nonlocal last_progress
                if kind == "created":
                    dump(out/"entry.json", {"entry_id":data["entry_id"],"created_at":datetime.now(timezone.utc).isoformat(),
                        "package_sha256":record["package"]["sha256"]})
                    print(json.dumps({"event":kind,"entry_id":data["entry_id"]}),flush=True)
                elif kind == "upload_progress" and time.monotonic()-last_progress >= 15:
                    last_progress = time.monotonic()
                    print(json.dumps({"event":kind,"done":data["done"],"total":data["total"]}),flush=True)
                elif kind in {"upload_start","launched","resume"}:
                    print(json.dumps({"event":kind,"entry_id":data.get("entry_id")}),flush=True)
            result = submit.run_submit(endpoint=ENDPOINT,token=token,path=record["package"]["path"],
                model_name=MODEL,description=DESCRIPTION,verify_targets=True,check_cell_counts=True,
                wait=False,force=False,skip_limit_check=False,resume=resume,
                pending_uploads=None if resume else auth.list_pending_uploads("default"),on_event=event,
                save_resume=lambda eid,data:auth.save_pending_upload("default",eid,data),
                clear_resume=lambda eid:auth.clear_pending_upload("default",eid))
            require(result.entry_id == safe.read_json(out/"entry.json")["entry_id"], "Submitted entry identity differs")
            dump(out/"submission.json",result.to_dict())
        finally:
            lock.release()
        return monitor(token,result.entry_id,out,seconds)
    except (api.ApiError,submit.SubmitError) as exc:
        # Do not expose signed upload URLs or credentials from exception details.
        print(json.dumps({"error_type":type(exc).__name__,"code":getattr(exc,"code",None),
            "new_entry_retry_performed":False,"inspect_private_resume_state":True}),flush=True)
        raise SystemExit(1) from None
    finally:
        del token


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["submit","resume","status"])
    for key in ("package-complete","package-sha","out"): p.add_argument("--"+key,required=True)
    p.add_argument("--monitor-seconds",type=int,default=1800)
    a = p.parse_args()
    print(json.dumps(execute(a.action,a.package_complete,a.package_sha,a.out,a.monitor_seconds),sort_keys=True),flush=True)


if __name__ == "__main__": main()
