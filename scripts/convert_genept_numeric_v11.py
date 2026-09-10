"""Pinned GenePT Ada archive -> numeric NPZ, with no pickle execution or fitting.

This is a data-only parser for an audited protocol-4 primitive grammar, NOT a
general pickle loader. Unknown opcodes are rejected BEFORE their arguments.
Public aliases are retained literally. Missing names have an explicit mask.
"""
from __future__ import annotations
import argparse
from array import array
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import struct
import time
import zipfile
import numpy as np

SCHEMA = "genept-numeric-conversion-v11"
ARCHIVE_SHA = "6193575dcbd7bbf214c8ca3eb518cb3d13272443f98ecd6c26402411a7ca745e"
ARCHIVE_BYTES, MEMBER_BYTES, DIM = 574395233, 460797248, 1536
MEMBER = "GenePT_emebdding_v2/GenePT_gene_embedding_ada_text.pickle"
TRAIN_SHA = "590a5916c6a9c76e263f2c8e4e45ed9ba41def7682a6694ce82d87b1b7074f2b"
TARGET_SHA = "f57edd7b912ebd718efc7ee9d0f334772513e7cc418d133ce525470e373b3276"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_regular(path, limit, *, payload=False):
    """Bounded same-handle read; do not follow leaf links or block on FIFOs."""
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit, "Regular bounded file required")
        sha, blocks, size = hashlib.sha256(), [], 0
        while True:
            block = stream.read(min(1 << 20, limit-size+1))
            if not block:
                break
            size += len(block)
            require(size <= limit, "File grew beyond bound")
            sha.update(block)
            if payload:
                blocks.append(block)
        after = os.fstat(stream.fileno())
        require(size == before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), "File changed during read")
        return b"".join(blocks) if payload else sha.hexdigest()


def digest(path):
    return read_regular(path, 128 << 20)


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh_json(path, value):
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def bound_json(path, sha):
    payload = read_regular(path, 2 << 20, payload=True)
    require(hashlib.sha256(payload).hexdigest() == sha, "JSON digest mismatch")
    return json.loads(payload)


def parse_vectors(payload, *, dimension=DIM, max_names=100000, max_vectors=34000,
                  max_memo=130000, max_bytes=MEMBER_BYTES):
    """Interpret primitive data only; preserve shared binary64 array objects."""
    require(type(payload) is bytes and 3 <= len(payload) <= max_bytes, "Payload size/type")
    require(type(dimension) is int and 1 <= dimension <= 4096, "Dimension bound")
    require(payload[:2] == b"\x80\x04", "Only protocol 4 supported")
    stack, memo, names, vectors = [], [], None, []
    marker = object()
    pos, frame_end, count = 2, None, Counter({"PROTO": 1})

    def take(size):
        nonlocal pos
        end = pos + size
        require(end <= len(payload) and (frame_end is None or end <= frame_end), "Truncated opcode/frame boundary")
        start, pos = pos, end
        return start

    while pos < len(payload):
        if frame_end == pos:
            frame_end = None
        code = payload[take(1)]
        if code == 0x95:
            require(frame_end is None, "Nested frame")
            length = struct.unpack_from("<Q", payload, take(8))[0]
            require(1 <= length <= 1 << 20 and pos + length <= len(payload), "Invalid frame size")
            frame_end = pos + length
            count["FRAME"] += 1
        elif code == 0x7d:
            require(names is None and not stack and not memo, "Only one root dictionary allowed")
            names = {}
            stack.append(names)
            count["EMPTY_DICT"] += 1
        elif code == 0x94:
            require(stack and (stack[-1] is names or type(stack[-1]) in (str, array)), "Unsafe memo type")
            require(len(memo) < max_memo, "Memo limit")
            memo.append(stack[-1])
            count["MEMOIZE"] += 1
        elif code == 0x28:
            stack.append(marker)
            count["MARK"] += 1
        elif code == 0x8c:
            size = payload[take(1)]
            start = take(size)
            value = payload[start:start + size].decode("utf-8", errors="strict")
            require(value and value == value.strip() and all(ch.isprintable() for ch in value), "Invalid name")
            stack.append(value)
            count["SHORT_BINUNICODE"] += 1
        elif code == 0x5d:
            require(len(vectors) < max_vectors, "Vector count limit")
            value = array("d")
            vectors.append(value)
            stack.append(value)
            count["EMPTY_LIST"] += 1
        elif code == 0x47:
            value = struct.unpack_from(">d", payload, take(8))[0]
            require(math.isfinite(value), "Nonfinite embedding")
            stack.append(value)
            count["BINFLOAT"] += 1
        elif code in (0x68, 0x6a):
            index = payload[take(1)] if code == 0x68 else struct.unpack_from("<I", payload, take(4))[0]
            require(index < len(memo), "Missing memo reference")
            require(type(memo[index]) in (str, array), "Root/cyclic reference forbidden")
            stack.append(memo[index])
            count["BINGET" if code == 0x68 else "LONG_BINGET"] += 1
        elif code in (0x65, 0x75):
            mark = next((i for i in range(len(stack)-1, -1, -1) if stack[i] is marker), -1)
            require(mark >= 1, "Missing container marker")
            container, items = stack[mark-1], stack[mark+1:]
            if code == 0x65:
                require(type(container) is array and items and all(type(v) is float for v in items), "Only float lists allowed")
                require(len(container) + len(items) <= dimension, "Vector dimension exceeded")
                container.extend(items)
                count["APPENDS"] += 1
            else:
                require(container is names and mark == 1 and items and len(items) % 2 == 0, "Invalid root items")
                for i in range(0, len(items), 2):
                    key, value = items[i:i+2]
                    require(type(key) is str and key not in names, "Duplicate or invalid name")
                    require(type(value) is array and len(value) == dimension, "Incorrect vector dimension")
                    require(len(names) < max_names, "Name count limit")
                    names[key] = value
                count["SETITEMS"] += 1
            del stack[mark:]
        elif code == 0x2e:
            require(len(stack) == 1 and stack[0] is names and names and pos == len(payload), "Invalid root/EOF at STOP")
            require(frame_end is None or frame_end == pos, "Incomplete final frame")
            require(all(len(v) == dimension for v in vectors), "Incomplete vector object")
            require({id(v) for v in names.values()} == {id(v) for v in vectors}, "Unreferenced vector object")
            count["STOP"] += 1
            return names, {"name_keys": len(names), "stored_vector_objects": len(vectors), "dimension": dimension,
                           "opcode_counts": dict(sorted(count.items()))}
        else:
            raise ValueError(f"Forbidden opcode 0x{code:02x}; no argument decoded")
        require(len(stack) <= 4096, "Stack limit")
    raise ValueError("Missing STOP")


def select_features(names, targets, dimension=DIM):
    require(type(targets) is list and targets and len(targets) <= 20000
            and all(type(t) is str and t and t == t.strip() for t in targets)
            and targets == sorted(set(targets)), "Sorted unique bounded target names required")
    embedding = np.zeros((len(targets), dimension), dtype=np.float32)
    present = np.zeros(len(targets), dtype=np.uint8)
    reference_ids = np.full(len(targets), -1, dtype=np.int32)
    original_hashes = np.full(len(targets), "", dtype="U64")
    aliases, max_error = {}, 0.
    for i, target in enumerate(targets):
        if target not in names:
            continue
        original = np.asarray(names[target], dtype=np.float64)
        require(original.shape == (dimension,) and np.isfinite(original).all(), "Invalid original vector")
        require(np.abs(original).max() <= np.finfo(np.float32).max, "float32 overflow")
        embedding[i] = original
        require(np.linalg.norm(embedding[i].astype(np.float64)) > 0, "Zero-length available embedding")
        max_error = max(max_error, float(np.abs(embedding[i].astype(np.float64)-original).max()))
        present[i] = 1
        reference_ids[i] = aliases.setdefault(id(names[target]), len(aliases))
        original_hashes[i] = hashlib.sha256(original.astype("<f8", copy=False).tobytes()).hexdigest()
    return {"target_ids": np.asarray(targets), "embeddings": embedding, "present": present,
            "source_vector_object_id": reference_ids, "source_vector_sha256": original_hashes}, {"selected": len(targets), "covered": int(present.sum()),
        "missing": [t for t, p in zip(targets, present) if not p], "max_float32_roundoff": max_error,
        "selected_stored_vector_objects": len(aliases), "normalization": "none; raw pretrained values",
        "missing_policy": "zero vector plus present=0; no alias guessing or mean imputation"}


def sharing_audit(arrays, roles, official):
    """Duplicate representations are review signals, not proven gene equivalence."""
    fit = {t for role in roles.values() for t in role["fit_targets"]}
    held = {t for role in roles.values() for t in role["tune_targets"]}
    official = set(official)
    result = {}
    for kind in ("source_object", "binary64_content", "float32_content"):
        groups = {}
        for i, target in enumerate(arrays["target_ids"].tolist()):
            if not arrays["present"][i]:
                continue
            key = (int(arrays["source_vector_object_id"][i]) if kind == "source_object" else
                   str(arrays["source_vector_sha256"][i]) if kind == "binary64_content" else
                   hashlib.sha256(arrays["embeddings"][i].astype("<f4", copy=False).tobytes()).hexdigest())
            groups.setdefault(key, []).append(target)
        duplicates = sorted([sorted(g) for g in groups.values() if len(g) > 1])
        result[kind] = {"duplicate_groups": duplicates,
            "cross_fit_held_groups": [g for g in duplicates if set(g) & fit and set(g) & held],
            "fit_official_groups": [g for g in duplicates if set(g) & fit and set(g) & official]}
    return {"interpretation": "representation sharing only; not proof of biological synonymy", **result}


def policy():
    return {"dimension": DIM, "expected_name_keys": 93800, "expected_stored_vectors": 33230,
        "address_space_limit": 3 << 30, "rss_limit": 2 << 30, "wall_seconds": 600,
        "output_limit": 128 << 20, "cpu_threads": 2, "network": False, "model_fitting": False,
        "challenge_expression": False, "pickle_execution": False}


def plan(archive, training_contract, targets_csv, protocol):
    require(not any(Path(p).is_symlink() for p in (archive,training_contract,targets_csv,protocol)), "Leaf symlinks forbidden")
    archive, training_contract, targets_csv, protocol = map(lambda p: Path(p).resolve(),
        (archive, training_contract, targets_csv, protocol))
    parent = bound_json(training_contract, TRAIN_SHA)
    payload = read_regular(targets_csv, 2 << 20, payload=True)
    require(hashlib.sha256(payload).hexdigest() == TARGET_SHA, "Official target metadata changed")
    reader = csv.DictReader(payload.decode("utf-8").splitlines())
    require(reader.fieldnames == ["target_gene"], "Official target schema")
    official = [r["target_gene"] for r in reader]
    require(len(official) == len(set(official)) == 300, "Official target roster")
    roles = parent["sources"]
    require(set(roles) == {"replogle_k562", "nadig_jurkat"}, "Public source roster")
    public = {t for role in roles.values() for key in ("fit_targets", "tune_targets") for t in role[key]}
    return {"schema": SCHEMA, "policy": policy(), "converter_sha256": digest(__file__),
        "archive": str(archive), "archive_sha256": ARCHIVE_SHA, "archive_bytes": ARCHIVE_BYTES,
        "member": MEMBER, "member_bytes": MEMBER_BYTES,
        "training_contract": {"path": str(training_contract), "sha256": TRAIN_SHA},
        "official_targets": {"path": str(targets_csv), "sha256": TARGET_SHA},
        "protocol": {"path": str(protocol), "sha256": digest(protocol)}, "source_roles": roles,
        "official_target_names": sorted(official), "selected_targets": sorted(public | set(official)),
        "expression_read": False}


def register(archive, training_contract, targets_csv, protocol, out):
    require(not os.path.lexists(out), "Fresh registration required")
    out = Path(out).resolve()
    require(not out.exists(), "Fresh registration required")
    record = {**plan(archive, training_contract, targets_csv, protocol), "registered_at": now()}
    out.mkdir(mode=0o700)
    fresh_json(out / "contract.json", record)
    return digest(out / "contract.json")


def runtime_guard():
    require(socket.gethostname().split(".")[0] in {"cbsuvlaminck3", "cbsuvlaminck6"}, "Approved compute host required; never head/login")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "GPU must be hidden for conversion")
    require(len(os.sched_getaffinity(0)) <= 2, "At most two CPU cores required")
    require(all(os.environ.get(k) == "2" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")), "Thread bounds required")


def convert(contract, contract_sha, out):
    runtime_guard()
    start = time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS, (3 << 30, 3 << 30))
    previous = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("Conversion wall cap")))
    signal.alarm(600)
    try:
        return _convert(contract, contract_sha, out, start)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def _convert(contract, contract_sha, out, start):
    record = bound_json(contract, contract_sha)
    expected = plan(record["archive"], record["training_contract"]["path"],
                    record["official_targets"]["path"], record["protocol"]["path"])
    require({k:v for k,v in record.items() if k != "registered_at"} == expected, "Registration reconstruction failed")
    require(not os.path.lexists(out), "Fresh conversion output required")
    out = Path(out).resolve()
    require(not out.exists(), "Fresh conversion output required")
    out.mkdir(mode=0o700)
    fresh_json(out / "started.json", {"schema": SCHEMA, "started_at": now(), "contract_sha256": contract_sha})
    try:
        with os.fdopen(os.open(record["archive"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
            before = os.fstat(stream.fileno())
            require(stat.S_ISREG(before.st_mode) and before.st_size == ARCHIVE_BYTES, "Archive size/type")
            require(hashlib.file_digest(stream, "sha256").hexdigest() == ARCHIVE_SHA, "Archive digest mismatch")
            hashed = os.fstat(stream.fileno())
            require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                    (hashed.st_dev,hashed.st_ino,hashed.st_size,hashed.st_mtime_ns,hashed.st_ctime_ns),
                    "Archive changed during authentication")
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                require(len(members) <= 64 and len({m.filename for m in members}) == len(members), "ZIP duplicate/size")
                member = archive.getinfo(MEMBER)
                require(member.file_size == MEMBER_BYTES and not member.flag_bits & 1
                        and not stat.S_ISLNK(member.external_attr >> 16), "Member type/size/encryption")
                with archive.open(member) as handle:
                    payload = handle.read(MEMBER_BYTES + 1)
                    require(len(payload) == MEMBER_BYTES, "Expanded member length")
                member_sha = hashlib.sha256(payload).hexdigest()
                names, audit = parse_vectors(payload)
                require(audit["name_keys"] == 93800 and audit["stored_vector_objects"] == 33230, "Official structure changed")
                arrays, selection = select_features(names, record["selected_targets"])
                del names, payload
            after = os.fstat(stream.fileno())
            require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                    (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), "Archive changed during read")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(peak <= policy()["rss_limit"], "RSS cap exceeded")
        with os.fdopen(os.open(out / "features.npz", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        require((out / "features.npz").stat().st_size < policy()["output_limit"], "Output cap exceeded")
        with np.load(out / "features.npz", allow_pickle=False) as replay:
            require(set(replay.files) == set(arrays) and all(np.array_equal(replay[k], v) and replay[k].dtype == v.dtype
                    for k, v in arrays.items()), "Numeric-only NPZ replay failed")
        index = {t: i for i, t in enumerate(record["selected_targets"])}
        coverage = {s: {role: {"total": len(targets), "covered": sum(int(arrays["present"][index[t]]) for t in targets)}
            for role, targets in roles.items()} for s, roles in record["source_roles"].items()}
        coverage["official"] = {"total": 300, "covered": sum(int(arrays["present"][index[t]]) for t in record["official_target_names"])}
        sharing = sharing_audit(arrays, record["source_roles"], record["official_target_names"])
        require(expected == plan(record["archive"],record["training_contract"]["path"],
                record["official_targets"]["path"],record["protocol"]["path"]), "Provenance changed during conversion")
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        require(peak <= policy()["rss_limit"] and time.monotonic()-start < 600, "Completion resource cap exceeded")
        receipt = {"schema": SCHEMA, "completed": True, "completed_at": now(), "contract_sha256": contract_sha,
            "archive_sha256": ARCHIVE_SHA, "member_sha256": member_sha, "structure": audit, "selection": selection,
            "coverage": coverage, "vector_sharing": sharing, "feature_path": str(out / "features.npz"), "feature_sha256": digest(out / "features.npz"),
            "feature_bytes": (out / "features.npz").stat().st_size, "elapsed_seconds": time.monotonic()-start,
            "peak_rss_bytes": peak, "model_fitting_performed": False, "pickle_execution": False,
            "challenge_expression_read": False, "paid_api_used": False, "network_used": False}
        receipt["provenance_reauthenticated_at_completion"] = True
        receipt["automatic_training_admission"] = False
        require(sum(p.stat().st_size for p in out.iterdir()) + len(json.dumps(receipt,indent=2).encode()) + 1
                <= policy()["output_limit"], "Cumulative output cap")
        fresh_json(out / "complete.json", receipt)
        require(time.monotonic()-start < 600 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024 <= policy()["rss_limit"], "Final resource cap")
        return receipt
    except Exception as exc:
        fresh_json(out / "failure.json", {"failed_at": now(), "contract_sha256": contract_sha,
            "error_type": type(exc).__name__, "reason": str(exc)})
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="action", required=True)
    r = sub.add_parser("register")
    for name in ("archive", "training-contract", "targets-csv", "protocol", "out"):
        r.add_argument("--" + name, required=True)
    c = sub.add_parser("convert")
    for name in ("contract", "contract-sha", "out"):
        c.add_argument("--" + name, required=True)
    args = vars(p.parse_args())
    action = args.pop("action")
    print(json.dumps(register(**args) if action == "register" else convert(**args), sort_keys=True), flush=True)
