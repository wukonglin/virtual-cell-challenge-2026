"""Bounded public GenePT archive download; never unpickle or admit features.

Published archive MD5/size: https://zenodo.org/records/10833191 .
Only ZIP directory metadata is inspected, never member payloads or pickle code.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import stat
import time
import urllib.parse
import urllib.request
import zipfile

URL = "https://zenodo.org/records/10833191/files/GenePT_emebdding_v2.zip?download=1"
METADATA_URL = "https://api.datacite.org/dois/10.5281/zenodo.10833191"
NAME = "GenePT_emebdding_v2.zip"
SIZE = 574_395_233
MD5 = "3f6ce4317e3a0091978ae5cb8fbf05a3"
MAX_BYTES = 650 << 20
MAX_SECONDS = 600


def now():
    return datetime.now(timezone.utc).isoformat()


def fresh_bytes(path, payload):
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def fresh_json(path, value):
    fresh_bytes(path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode())


class OfficialOnlyRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, url):
        parsed = urllib.parse.urlparse(url)
        if (parsed.scheme != "https" or parsed.hostname not in {"zenodo.org", "api.datacite.org"}
                or parsed.username or parsed.password):
            raise ValueError("Refusing redirect outside official HTTPS metadata/archive hosts")
        return super().redirect_request(request, response, code, message, headers, url)


def license_from_metadata(record):
    attrs = record["data"]["attributes"]
    if attrs.get("doi") != "10.5281/zenodo.10833191":
        raise ValueError("Wrong official GenePT DOI")
    rights = attrs.get("rightsList", [])
    if not any(r.get("rightsIdentifier") == "cc-by-4.0" and
               r.get("rightsUri") == "https://creativecommons.org/licenses/by/4.0/legalcode" for r in rights):
        raise ValueError("Expected CC BY 4.0 license missing; review required")
    return rights


def zip_metadata(path):
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        if not 1 <= len(members) <= 64 or len({m.filename for m in members}) != len(members):
            raise ValueError("Unexpected duplicate/excessive ZIP members")
        if sum(m.file_size for m in members) > 4 << 30:
            raise ValueError("Declared ZIP expansion exceeds 4 GiB")
        result = []
        for member in members:
            name = Path(member.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in member.filename or member.flag_bits & 1
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise ValueError("Unsafe ZIP member path/type/encryption")
            result.append({"name": member.filename, "compressed_bytes": member.compress_size,
                "expanded_bytes": member.file_size, "crc32": f"{member.CRC:08x}", "directory": member.is_dir()})
    return result


def timeout_handler(signum, frame):
    raise TimeoutError("GenePT acquisition exceeded its 10 minute total wall cap")


def acquire(destination):
    destination = Path(destination).absolute()
    if any(part.is_symlink() for part in [destination, *destination.parents]):
        raise ValueError("Symlink destination components forbidden")
    if destination.exists():
        raise FileExistsError("Fresh acquisition directory required; inspect prior attempt instead of overwriting")
    destination.mkdir(mode=0o700, parents=True)
    started, received = time.monotonic(), 0
    fresh_json(destination / "started.json", {"schema": "genept-public-acquisition-v1", "started_at": now(),
        "url": URL, "license_metadata_url": METADATA_URL, "expected_bytes": SIZE, "published_md5": MD5,
        "transfer_cap_bytes": MAX_BYTES, "wall_seconds_cap": MAX_SECONDS, "gcp_used": False,
        "paid_api_used": False, "credentials_used": False, "unpickle_authorized": False,
        "features_admitted_to_training": False})
    opener = urllib.request.build_opener(OfficialOnlyRedirect())
    partial = destination / (NAME + ".part")
    prior_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(MAX_SECONDS)
    try:
        with opener.open(METADATA_URL, timeout=40) as response:
            payload = response.read((2 << 20) + 1)
            if len(payload) > 2 << 20:
                raise ValueError("Official license metadata exceeds 2 MiB")
        rights = license_from_metadata(json.loads(payload))
        fresh_bytes(destination / "official_datacite_metadata.json", payload)
        md5, sha = hashlib.md5(), hashlib.sha256()
        with opener.open(URL, timeout=40) as response:
            declared = response.headers.get("Content-Length")
            if response.status != 200 or declared is not None and int(declared) != SIZE:
                raise ValueError("Archive response status/length differs from registered source")
            with os.fdopen(os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as stream:
                reported = 0
                while True:
                    block = response.read(min(1 << 20, MAX_BYTES - received + 1))
                    if not block:
                        break
                    received += len(block)
                    if received > MAX_BYTES or received > SIZE:
                        raise ValueError("GenePT response exceeds exact size/campaign cap")
                    stream.write(block)
                    md5.update(block)
                    sha.update(block)
                    if received - reported >= 64 << 20:
                        print(json.dumps({"stage": "download", "received_bytes": received, "elapsed_seconds": round(time.monotonic() - started, 1)}), flush=True)
                        reported = received
                stream.flush()
                os.fsync(stream.fileno())
        if received != SIZE or md5.hexdigest() != MD5:
            raise ValueError("GenePT exact length or published MD5 verification failed")
        members = zip_metadata(partial)
        final = destination / NAME
        # Atomic no-overwrite publication. Retain the partial as a hardlink to
        # the same inode; this does not allocate a second archive copy.
        os.link(partial, final, follow_symlinks=False)
        receipt = {"schema": "genept-public-acquisition-v1", "status": "archive_verified_not_features",
            "completed_at": now(), "archive": str(final), "archive_bytes": received,
            "published_md5": MD5, "sha256": sha.hexdigest(), "url": URL,
            "checksum_source": "https://zenodo.org/records/10833191", "doi": "10.5281/zenodo.10833191",
            "license_metadata_url": METADATA_URL, "license": rights,
            "metadata_sha256": hashlib.sha256(payload).hexdigest(),
            "elapsed_seconds": time.monotonic() - started, "zip_members": members,
            "zip_member_payloads_read": False, "unpickled": False, "features_admitted_to_training": False,
            "gcp_used": False, "paid_api_used": False, "credentials_used": False,
            "partial_retained_as_hardlink": str(partial)}
        fresh_json(destination / "complete.json", receipt)
        print(json.dumps(receipt, sort_keys=True), flush=True)
        return receipt
    except Exception as exc:
        signal.alarm(0)
        fresh_json(destination / "failure.json", {"schema": "genept-public-acquisition-v1", "failed_at": now(),
            "received_bytes": received, "elapsed_seconds": time.monotonic() - started,
            "error_type": type(exc).__name__, "reason": str(exc), "verified_archive": False})
        raise
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prior_handler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    acquire(parser.parse_args().destination)
