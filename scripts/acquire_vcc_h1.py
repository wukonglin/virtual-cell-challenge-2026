"""Acquire exactly seven processed H1 objects via the official Google CLI.

No recursive media transfer, FASTQ, automatic retry, overwrite, or expression
inspection. The user-confirmed Marketplace allowance is not a Google hard cap.
A project/month ledger conservatively reserves the whole object before each
attempt, including failed transfers. Independent project activity is not metered.
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

PROJECT = "vcc-dataset"
PREFIX = "gs://arc-institute-virtual-cell-atlas/virtual-cell-challenge/2025/"
MONTHLY_BYTES = 2_000_000_000_000
CAMPAIGN_BYTES = 50_000_000_000
SAFETY_BYTES = 100_000_000_000
NAMES = frozenset({"gene_names.csv", "train/adata_Training.h5ad",
                   "train/pert_counts_Training.csv", "validation/adata_Validation.h5ad",
                   "validation/pert_counts_Validation.csv", "test/adata_Test.h5ad",
                   "test/pert_counts_Test.csv"})


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _digest(value, size):
    if not isinstance(value, str):
        raise ValueError("Missing upstream checksum")
    try:
        raw = base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Malformed upstream checksum") from exc
    if len(raw) != size:
        raise ValueError("Wrong checksum width")


def validate_manifest(manifest):
    if manifest.get("schema") != "vcc-h1-acquisition-plan-v1" or manifest.get("billing_project") != PROJECT:
        raise ValueError("Wrong schema/billing project")
    if manifest.get("user_confirmed_subscription") is not True:
        raise ValueError("Marketplace confirmation required")
    if manifest.get("allowance_month") != dt.datetime.now(dt.timezone.utc).strftime("%Y-%m"):
        raise ValueError("Allowance confirmation is for a different month")
    for key, value in (("allowance_bytes", MONTHLY_BYTES), ("campaign_cap_bytes", CAMPAIGN_BYTES),
                       ("safety_reserve_bytes", SAFETY_BYTES)):
        if type(manifest.get(key)) is not int or manifest[key] != value:
            raise ValueError(f"Unsupported {key}")
    prior = manifest.get("prior_usage_bytes")
    if type(prior) is not int or not 0 <= prior <= MONTHLY_BYTES:
        raise ValueError("Invalid prior usage")
    objects = manifest.get("objects")
    if not isinstance(objects, list) or len(objects) != len(NAMES):
        raise ValueError("Exactly seven processed H1 objects are required")
    if any(not isinstance(obj, dict) for obj in objects):
        raise ValueError("Invalid object metadata")
    if any(not isinstance(obj.get("name"), str) for obj in objects) or {obj.get("name") for obj in objects} != NAMES:
        raise ValueError("Only exact processed H1 names are allowed; no FASTQ or path traversal")
    for obj in objects:
        if not isinstance(obj.get("generation"), str) or re.fullmatch(r"[1-9][0-9]*", obj["generation"]) is None:
            raise ValueError("Immutable object generation required")
        if type(obj.get("size_bytes")) is not int or obj["size_bytes"] <= 0:
            raise ValueError("Invalid object size")
        if obj.get("storage_class") != "STANDARD" or obj.get("content_encoding") is not None:
            raise ValueError("Unexpected storage class/encoding; reassess charges and byte accounting")
        _digest(obj.get("crc32c"), 4)
        if obj.get("md5") is not None:
            _digest(obj["md5"], 16)
    total = sum(obj["size_bytes"] for obj in objects)
    if total != manifest.get("planned_bytes") or total > CAMPAIGN_BYTES:
        raise ValueError("Manifest size/campaign cap mismatch")
    if prior + total + SAFETY_BYTES > MONTHLY_BYTES:
        raise ValueError("Insufficient confirmed monthly allowance")
    return objects


def safe_destination(root, name):
    if name not in NAMES:
        raise ValueError("Unexpected destination")
    if root.is_symlink():
        raise ValueError("Symlink destination root")
    current = root
    for part in Path(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("Symlink destination component")
    return current


def verify_file(path, obj):
    import google_crc32c
    if path.is_symlink() or not path.is_file() or path.stat().st_size != obj["size_bytes"]:
        raise ValueError("Existing/downloaded file is not a regular file of the pinned size")
    crc = google_crc32c.Checksum()
    sha, md5 = hashlib.sha256(), hashlib.md5()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            crc.update(block)
            sha.update(block)
            md5.update(block)
    observed_crc = base64.b64encode(crc.digest()).decode()
    observed_md5 = base64.b64encode(md5.digest()).decode()
    if observed_crc != obj["crc32c"] or (obj.get("md5") is not None and observed_md5 != obj["md5"]):
        raise ValueError("Upstream checksum mismatch; file not accepted")
    return {"name": obj["name"], "generation": obj["generation"], "size_bytes": obj["size_bytes"],
            "crc32c": observed_crc, "md5": observed_md5, "sha256": sha.hexdigest()}


def gcloud_command(wrapper, obj, partial, csv_log):
    return ["bash", str(wrapper), "storage", "cp", "--billing-project", PROJECT,
            "--do-not-decompress", "--manifest-path", str(csv_log),
            PREFIX + obj["name"] + "#" + obj["generation"], str(partial)]


def read_ledger(path, month):
    if path.is_symlink():
        raise ValueError("Symlink ledger forbidden")
    events = []
    if path.exists():
        for line in path.read_text().splitlines():
            event = json.loads(line)
            if event.get("project") != PROJECT or event.get("month") != month:
                raise ValueError("Ledger identity mismatch")
            if event.get("event") == "reserve" and (type(event.get("bytes")) is not int or event["bytes"] <= 0):
                raise ValueError("Invalid reserved bytes")
            if event.get("event") not in {"reserve", "verified", "failed"}:
                raise ValueError("Unexpected ledger event")
            events.append(event)
    return events


def reserved_bytes(events):
    return sum(event["bytes"] for event in events if event["event"] == "reserve")


def check_reservation(manifest, events, additional):
    if type(additional) is not int or additional < 0:
        raise ValueError("Additional reservation must be a nonnegative integer")
    total = reserved_bytes(events) + additional
    if total > CAMPAIGN_BYTES:
        raise ValueError("50 GB campaign cap exceeded (failed attempts remain reserved)")
    if manifest["prior_usage_bytes"] + total + SAFETY_BYTES > MONTHLY_BYTES:
        raise ValueError("Monthly allowance minus safety margin would be exceeded")


def append_event(path, events, event):
    # Exclusive project/month flock is held by the caller for the whole job.
    flags = os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "a") as stream:
        stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    events.append(event)


def publish_json(path, payload):
    # No overwrite: a failed run is resumed from its ledger, not a stale success.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
    with os.fdopen(os.open(path, flags, 0o600), "w") as stream:
        json.dump(payload, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ledger-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    os.umask(0o077)
    payload = args.manifest.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != args.manifest_sha256:
        raise ValueError("Manifest SHA-256 mismatch")
    manifest = json.loads(payload)
    objects = validate_manifest(manifest)
    root = args.destination.absolute()
    if root.is_symlink() or not root.is_dir():
        raise ValueError("Existing regular destination directory required")
    args.ledger_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    if args.ledger_dir.is_symlink():
        raise ValueError("Symlink ledger directory forbidden")
    month = manifest["allowance_month"]
    lock_path = args.ledger_dir / f"{PROJECT}-{month}.lock"
    ledger_path = args.ledger_dir / f"{PROJECT}-{month}.jsonl"
    wrapper = Path(__file__).resolve().with_name("vcc_gcloud.sh")
    receipts = []
    with os.fdopen(os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600), "a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        events = read_ledger(ledger_path, month)
        if (root / "acquisition_complete.json").exists():
            raise ValueError("Completion receipt already exists; use local verification, not another download")
        if (root / "gcloud_transfers.csv").is_symlink():
            raise ValueError("Symlink transfer log forbidden")
        remaining = sum(obj["size_bytes"] for obj in objects if not safe_destination(root, obj["name"]).exists())
        check_reservation(manifest, events, remaining)
        if shutil.disk_usage(root).free < remaining + 1_000_000_000:
            raise ValueError("Insufficient local disk space")
        for obj in objects:
            destination = safe_destination(root, obj["name"])
            destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if destination.exists():
                receipts.append({**verify_file(destination, obj), "action": "verified_existing"})
                print(json.dumps({"status": "verified_existing", "name": obj["name"]}), flush=True)
                continue
            partial = destination.with_name(destination.name + "." + obj["generation"] + ".part")
            sidecar = Path(str(partial) + "_.gstmp")
            if any(event["event"] == "reserve" and event.get("name") == obj["name"] for event in events):
                raise ValueError("Prior uncompleted reservation found; manual review required, no automatic retry")
            if partial.exists() or partial.is_symlink() or sidecar.exists() or sidecar.is_symlink():
                raise ValueError("Orphan partial found; no automatic resume/retry or overwrite")
            check_reservation(manifest, events, obj["size_bytes"])
            identity = {"project": PROJECT, "month": month, "manifest_sha256": digest,
                        "name": obj["name"], "generation": obj["generation"]}
            append_event(ledger_path, events, {**identity, "event": "reserve", "bytes": obj["size_bytes"], "at": utc_now()})
            print(json.dumps({"status": "downloading", "name": obj["name"], "size_bytes": obj["size_bytes"],
                              "campaign_reserved_bytes": reserved_bytes(events)}), flush=True)
            try:
                subprocess.run(gcloud_command(wrapper, obj, partial, root / "gcloud_transfers.csv"), check=True)
                receipt = verify_file(partial, obj)
                os.link(partial, destination, follow_symlinks=False)
                partial.unlink()  # Same verified bytes remain at the no-clobber destination.
                receipts.append({**receipt, "action": "downloaded_verified"})
                append_event(ledger_path, events, {**identity, "event": "verified", "at": utc_now(), **receipt})
                print(json.dumps({"status": "verified", **receipt}), flush=True)
            except BaseException:
                append_event(ledger_path, events, {**identity, "event": "failed", "at": utc_now()})
                raise
        publish_json(root / "acquisition_complete.json", {
            "schema": "vcc-h1-acquisition-complete-v1", "status": "complete", "completed_at": utc_now(),
            "billing_project": PROJECT, "manifest_sha256": digest, "files": receipts,
            "verified_bytes": sum(row["size_bytes"] for row in receipts),
            "campaign_reserved_bytes": reserved_bytes(events), "campaign_cap_bytes": CAMPAIGN_BYTES,
            "raw_fastq_downloaded": False, "expression_inspected": False, "training_started": False,
            "allowance_basis": "user-confirmed Marketplace subscription and zero prior monthly usage",
            "billing_credit_independently_verified": False,
        })
        print(json.dumps({"status": "complete", "files": len(receipts), "verified_bytes": manifest["planned_bytes"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
