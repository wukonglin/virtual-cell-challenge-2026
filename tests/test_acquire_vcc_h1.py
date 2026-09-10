"""Offline acquisition-contract tests. All transferred bytes are synthetic."""
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import google_crc32c
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import acquire_vcc_h1 as h1  # noqa: E402


def object_record(name, payload=b"synthetic counts placeholder\n"):
    crc = google_crc32c.Checksum(payload)
    return dict(name=name, generation="1234567890", size_bytes=len(payload),
                crc32c=base64.b64encode(crc.digest()).decode(),
                md5=base64.b64encode(hashlib.md5(payload).digest()).decode(),
                storage_class="STANDARD", content_encoding=None)


@pytest.fixture
def manifest():
    objects = [object_record(name) for name in sorted(h1.NAMES)]
    return dict(schema="vcc-h1-acquisition-plan-v1", billing_project=h1.PROJECT,
                user_confirmed_subscription=True,
                allowance_month=h1.dt.datetime.now(h1.dt.timezone.utc).strftime("%Y-%m"),
                allowance_bytes=h1.MONTHLY_BYTES, campaign_cap_bytes=h1.CAMPAIGN_BYTES,
                safety_reserve_bytes=h1.SAFETY_BYTES, prior_usage_bytes=0,
                planned_bytes=sum(obj["size_bytes"] for obj in objects), objects=objects)


@pytest.fixture(autouse=True)
def no_real_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected real process/cloud request in an offline test")
    monkeypatch.setattr(h1.subprocess, "run", forbidden)
    previous_umask = os.umask(0o077)
    os.umask(previous_umask)
    yield
    os.umask(previous_umask)


def test_valid_manifest_and_immutable_approved_names(manifest):
    result = h1.validate_manifest(manifest)
    assert len(result) == 7
    assert {obj["name"] for obj in result} == h1.NAMES
    assert all("fastq" not in obj["name"].lower() for obj in result)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "train/../gene_names.csv",
                                  "train/x.fastq.gz", "train/adata_Training.h5ad#1",
                                  "train/adata_Training.h5ad\nother", "gs://another-bucket/x", None])
def test_manifest_rejects_fastq_traversal_and_unregistered_names(manifest, name):
    manifest["objects"][0]["name"] = name
    with pytest.raises(ValueError):
        h1.validate_manifest(manifest)


@pytest.mark.parametrize("key,value", [
    ("generation", None), ("generation", 123), ("generation", "0"),
    ("generation", "01"), ("generation", "1#2"), ("generation", "1\n"),
    ("size_bytes", True), ("size_bytes", 1.0), ("size_bytes", 0), ("size_bytes", -1),
    ("crc32c", "not-base64"), ("crc32c", None), ("crc32c", "YQ=="),
    ("crc32c", ""), ("md5", "YQ=="), ("md5", "!"),
    ("storage_class", "NEARLINE"), ("content_encoding", "gzip"),
])
def test_manifest_rejects_bad_object_metadata(manifest, key, value):
    manifest["objects"][0][key] = value
    with pytest.raises(ValueError):
        h1.validate_manifest(manifest)


@pytest.mark.parametrize("changes", [
    {"schema": "other"}, {"billing_project": "other-project"},
    {"user_confirmed_subscription": False}, {"user_confirmed_subscription": 1},
    {"allowance_month": "1900-01"}, {"allowance_bytes": True},
    {"campaign_cap_bytes": 60_000_000_000}, {"safety_reserve_bytes": 0},
    {"prior_usage_bytes": True}, {"prior_usage_bytes": -1},
    {"prior_usage_bytes": h1.MONTHLY_BYTES}, {"planned_bytes": 0},
    {"objects": []}, {"objects": [None] * 7},
])
def test_manifest_rejects_bad_envelope_and_budget(manifest, changes):
    manifest.update(changes)
    with pytest.raises(ValueError):
        h1.validate_manifest(manifest)


def test_duplicate_missing_and_eighth_object_rejected(manifest):
    for objects in (manifest["objects"][:-1], manifest["objects"] + [manifest["objects"][0]],
                    [manifest["objects"][0]] * 7):
        other = copy.deepcopy(manifest)
        other["objects"] = objects
        with pytest.raises(ValueError):
            h1.validate_manifest(other)


def test_campaign_cap_checked_against_total(manifest):
    manifest["objects"][0]["size_bytes"] = h1.CAMPAIGN_BYTES
    manifest["planned_bytes"] = sum(obj["size_bytes"] for obj in manifest["objects"])
    with pytest.raises(ValueError, match="cap"):
        h1.validate_manifest(manifest)


def test_verify_file_checks_upstream_and_records_local_digest(tmp_path):
    payload = b"synthetic data, no actual gene expression\n"
    path = tmp_path / "synthetic.h5ad"
    path.write_bytes(payload)
    obj = object_record("train/adata_Training.h5ad", payload)
    receipt = h1.verify_file(path, obj)
    assert receipt["sha256"] == hashlib.sha256(payload).hexdigest()
    assert receipt["crc32c"] == obj["crc32c"]
    assert receipt["generation"] == obj["generation"]
    assert receipt["size_bytes"] == len(payload)
    obj["md5"] = None
    assert h1.verify_file(path, obj)["sha256"] == receipt["sha256"]


@pytest.mark.parametrize("failure", ["same-size-corrupt", "wrong-size", "bad-md5", "bad-crc", "symlink", "directory", "absent"])
def test_verify_file_rejects_untrusted_local_bytes(tmp_path, failure):
    payload = b"synthetic"
    obj = object_record("gene_names.csv", payload)
    path = tmp_path / "input"
    if failure == "directory":
        path.mkdir()
    elif failure == "symlink":
        target = tmp_path / "target"
        target.write_bytes(payload)
        path.symlink_to(target)
    elif failure != "absent":
        path.write_bytes(b"Synthetic" if failure == "same-size-corrupt" else payload)
        if failure == "wrong-size":
            path.write_bytes(payload + b"x")
        elif failure == "bad-md5":
            obj["md5"] = base64.b64encode(b"0" * 16).decode()
        elif failure == "bad-crc":
            obj["crc32c"] = base64.b64encode(b"0" * 4).decode()
    with pytest.raises(ValueError):
        h1.verify_file(path, obj)


def test_destination_rejects_root_parent_component_and_leaf_symlinks(tmp_path):
    root, elsewhere = tmp_path / "root", tmp_path / "elsewhere"
    root.mkdir()
    elsewhere.mkdir()
    assert h1.safe_destination(root, "gene_names.csv") == root / "gene_names.csv"
    for name in ("../outside", "train/x.fastq", "/tmp/absolute"):
        with pytest.raises(ValueError):
            h1.safe_destination(root, name)
    root_link = tmp_path / "root-link"
    root_link.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        h1.safe_destination(root_link, "gene_names.csv")
    (root / "train").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        h1.safe_destination(root, "train/adata_Training.h5ad")
    (root / "gene_names.csv").symlink_to(elsewhere / "absent")
    with pytest.raises(ValueError):
        h1.safe_destination(root, "gene_names.csv")


def test_command_binds_billing_generation_and_exact_source(tmp_path):
    obj = object_record("train/adata_Training.h5ad")
    wrapper, partial, log = (tmp_path / name for name in ("wrapper with spaces.sh", "partial", "transfers.csv"))
    command = h1.gcloud_command(wrapper, obj, partial, log)
    assert command == ["bash", str(wrapper), "storage", "cp", "--billing-project", "vcc-dataset",
                       "--do-not-decompress", "--manifest-path", str(log),
                       h1.PREFIX + obj["name"] + "#1234567890", str(partial)]
    assert not any(arg in command for arg in ("--recursive", "-r", "--no-clobber"))


def reserve(manifest, amount, name="gene_names.csv"):
    return dict(project=h1.PROJECT, month=manifest["allowance_month"], event="reserve",
                bytes=amount, name=name, generation="1234567890", manifest_sha256="a" * 64)


def test_ledger_reservations_not_refunded_by_failure_or_success(tmp_path, manifest):
    events = [reserve(manifest, 11), {**reserve(manifest, 11), "event": "failed"},
              reserve(manifest, 17), {**reserve(manifest, 17), "event": "verified"}]
    path = tmp_path / "ledger.jsonl"
    recorded = []
    for event in events:
        h1.append_event(path, recorded, event)
    assert recorded == events
    assert h1.read_ledger(path, manifest["allowance_month"]) == events
    assert h1.reserved_bytes(events) == 28
    assert h1.reserved_bytes(h1.read_ledger(path, manifest["allowance_month"])) == 28


@pytest.mark.parametrize("changes", [{"project": "other"}, {"month": "1900-01"},
                                       {"bytes": -1}, {"bytes": True}, {"bytes": 1.5},
                                       {"event": "refund"}])
def test_ledger_rejects_identity_drift_and_bad_events(tmp_path, manifest, changes):
    event = reserve(manifest, 12)
    event.update(changes)
    path = tmp_path / "ledger"
    path.write_text(json.dumps(event) + "\n")
    with pytest.raises(ValueError):
        h1.read_ledger(path, manifest["allowance_month"])


def test_ledger_rejects_symlink_and_truncated_record(tmp_path, manifest):
    path = tmp_path / "ledger"
    path.write_text('{"project":')
    with pytest.raises(ValueError):
        h1.read_ledger(path, manifest["allowance_month"])
    link = tmp_path / "link"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        h1.read_ledger(link, manifest["allowance_month"])


def test_reservation_caps_include_prior_attempts_and_safety_margin(manifest):
    events = [reserve(manifest, h1.CAMPAIGN_BYTES - 5)]
    h1.check_reservation(manifest, events, 5)
    with pytest.raises(ValueError, match="campaign"):
        h1.check_reservation(manifest, events, 6)
    manifest["prior_usage_bytes"] = h1.MONTHLY_BYTES - h1.SAFETY_BYTES - 5
    h1.check_reservation(manifest, [], 5)
    with pytest.raises(ValueError, match="Monthly"):
        h1.check_reservation(manifest, [], 6)
    for invalid in (-1, True, 1.5):
        with pytest.raises(ValueError):
            h1.check_reservation(manifest, [], invalid)


def campaign(tmp_path, manifest):
    root = tmp_path / "h1"
    root.mkdir()
    path = tmp_path / "plan.json"
    payload = json.dumps(manifest, sort_keys=True).encode()
    path.write_bytes(payload)
    ledger_dir = tmp_path / "ledger"
    args = ["--manifest", str(path), "--manifest-sha256", hashlib.sha256(payload).hexdigest(),
            "--destination", str(root), "--ledger-dir", str(ledger_dir)]
    ledger = ledger_dir / f"{h1.PROJECT}-{manifest['allowance_month']}.jsonl"
    return root, ledger, args


def fake_success(monkeypatch, manifest, ledger, calls):
    def run(command, check):
        assert check is True
        source = command[-2]
        name = source.removeprefix(h1.PREFIX).split("#")[0]
        events = h1.read_ledger(ledger, manifest["allowance_month"])
        assert events[-1]["event"] == "reserve" and events[-1]["name"] == name
        calls.append(command)
        Path(command[-1]).write_bytes(b"synthetic counts placeholder\n")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(h1.subprocess, "run", run)


def test_main_success_reserves_before_fake_transfer_and_verifies_all(tmp_path, manifest, monkeypatch):
    root, ledger, args = campaign(tmp_path, manifest)
    calls = []
    fake_success(monkeypatch, manifest, ledger, calls)
    assert h1.main(args) == 0
    assert len(calls) == 7
    receipt = json.loads((root / "acquisition_complete.json").read_text())
    assert receipt["verified_bytes"] == manifest["planned_bytes"]
    assert receipt["campaign_reserved_bytes"] == manifest["planned_bytes"]
    assert receipt["expression_inspected"] is False
    assert all(row["action"] == "downloaded_verified" for row in receipt["files"])
    events = h1.read_ledger(ledger, manifest["allowance_month"])
    assert [event["event"] for event in events] == ["reserve", "verified"] * 7
    assert not list(root.rglob("*.part"))
    with pytest.raises(ValueError, match="Completion receipt"):
        h1.main(args)
    assert len(calls) == 7


def test_main_authenticated_existing_files_skip_all_transfer(tmp_path, manifest):
    root, ledger, args = campaign(tmp_path, manifest)
    for obj in manifest["objects"]:
        path = root / obj["name"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic counts placeholder\n")
    assert h1.main(args) == 0
    receipt = json.loads((root / "acquisition_complete.json").read_text())
    assert receipt["campaign_reserved_bytes"] == 0
    assert all(row["action"] == "verified_existing" for row in receipt["files"])
    assert not ledger.exists()


def test_main_failure_keeps_reservation_and_refuses_automatic_retry(tmp_path, manifest, monkeypatch):
    root, ledger, args = campaign(tmp_path, manifest)
    calls = []
    def fail(command, check):
        calls.append(command)
        raise subprocess.CalledProcessError(1, command)
    monkeypatch.setattr(h1.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        h1.main(args)
    events = h1.read_ledger(ledger, manifest["allowance_month"])
    assert [event["event"] for event in events] == ["reserve", "failed"]
    assert h1.reserved_bytes(events) == manifest["objects"][0]["size_bytes"]
    with pytest.raises(ValueError, match="Prior uncompleted reservation"):
        h1.main(args)
    assert len(calls) == 1
    assert not (root / "acquisition_complete.json").exists()


@pytest.mark.parametrize("kind", ["part", "gstmp", "part-symlink", "log-symlink", "bad-existing", "prior-reserve"])
def test_main_existing_conflicts_fail_before_transfer(tmp_path, manifest, kind):
    root, ledger, args = campaign(tmp_path, manifest)
    obj = manifest["objects"][0]
    destination = root / obj["name"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + "." + obj["generation"] + ".part")
    if kind == "part":
        partial.write_bytes(b"partial")
    elif kind == "gstmp":
        Path(str(partial) + "_.gstmp").write_bytes(b"partial")
    elif kind == "part-symlink":
        partial.symlink_to(tmp_path / "absent")
    elif kind == "log-symlink":
        (root / "gcloud_transfers.csv").symlink_to(tmp_path / "absent")
    elif kind == "bad-existing":
        destination.write_bytes(b"corrupted counts placeholder\n")
    elif kind == "prior-reserve":
        ledger.parent.mkdir()
        ledger.write_text(json.dumps(reserve(manifest, obj["size_bytes"], obj["name"])) + "\n")
    with pytest.raises(ValueError):
        h1.main(args)
    assert not (root / "acquisition_complete.json").exists()


def test_main_pinned_manifest_hash_and_disk_guards(tmp_path, manifest, monkeypatch):
    root, ledger, args = campaign(tmp_path, manifest)
    wrong = list(args)
    wrong[wrong.index("--manifest-sha256") + 1] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        h1.main(wrong)
    monkeypatch.setattr(h1.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError, match="disk"):
        h1.main(args)
    assert not ledger.exists()
    assert not (root / "acquisition_complete.json").exists()


def test_publish_json_never_overwrites(tmp_path):
    path = tmp_path / "receipt.json"
    h1.publish_json(path, {"status": "complete"})
    with pytest.raises(FileExistsError):
        h1.publish_json(path, {"status": "replacement"})
    assert json.loads(path.read_text()) == {"status": "complete"}
