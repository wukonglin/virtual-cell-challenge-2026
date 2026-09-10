"""One bounded official-API GenePT retry; preserve the previous failed attempt.

Authenticate the official public file record, then reuse the existing exact-size,
MD5, SHA256 and ZIP-directory-only acquisition. Never unpickle or admit features.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import urllib.request

import acquire_genept_v2 as bounded

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "dataset/reference/genept/10833191_api_retry_20260910"
RECORD_URL = "https://zenodo.org/api/records/10833191"
CONTENT_URL = "https://zenodo.org/api/records/10833191/files/GenePT_emebdding_v2.zip/content"
MAX_SECONDS = 600


def validate_record(record):
    if record.get("id") != 10833191 or record.get("doi") != "10.5281/zenodo.10833191":
        raise ValueError("Exact official GenePT record identity required")
    files = record.get("files")
    if not isinstance(files, list) or len(files) != 1:
        raise ValueError("Exact one-file official GenePT record required")
    entry = files[0]
    if (entry.get("key") != bounded.NAME or entry.get("size") != bounded.SIZE
            or entry.get("checksum") != "md5:" + bounded.MD5
            or entry.get("links", {}).get("self") != CONTENT_URL):
        raise ValueError("Official GenePT filename, size, published MD5 or file URL differs")
    return entry


def acquire(destination):
    destination = Path(destination).absolute()
    if destination != DESTINATION or destination.exists():
        raise ValueError("Exact fresh, physical API retry directory required")
    started = time.monotonic()
    code_paths = (Path(__file__), Path(bounded.__file__))
    codes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in code_paths}
    opener = urllib.request.build_opener(bounded.OfficialOnlyRedirect())
    with opener.open(urllib.request.Request(RECORD_URL, headers={"Accept": "application/json"}), timeout=35) as response:
        if response.status != 200:
            raise ValueError("Official GenePT record did not return HTTP200")
        payload = response.read((1 << 20) + 1)
        if len(payload) > 1 << 20:
            raise ValueError("Official record exceeds 1 MiB metadata cap")
    entry = validate_record(json.loads(payload))
    remaining = MAX_SECONDS - math.ceil(time.monotonic() - started)
    if remaining <= 0:
        raise TimeoutError("GenePT total 600-second retry deadline exceeded")
    # Only this isolated process changes the legacy module's URL/time parameters;
    # its source and all records from the first failed attempt remain untouched.
    bounded.URL = CONTENT_URL
    bounded.MAX_SECONDS = remaining
    result = None
    try:
        result = bounded.acquire(destination)
        return result
    finally:
        if destination.is_dir() and not destination.is_symlink():
            bounded.fresh_bytes(destination / "official_zenodo_api_record.json", payload)
            bounded.fresh_json(destination / "api_retry_provenance.json", {
                "schema": "genept-public-api-retry-v1", "record_url": RECORD_URL,
                "record_sha256": hashlib.sha256(payload).hexdigest(),
                "verified_file": entry, "archive_url": CONTENT_URL,
                "code_sha256": codes, "total_wall_seconds_cap": MAX_SECONDS,
                "elapsed_seconds": time.monotonic() - started,
                "archive_verified": result is not None,
                "gcp_used": False, "paid_api_used": False, "credentials_used": False,
                "unpickled": False, "features_admitted_to_training": False,
                "previous_attempt_preserved": True,
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    acquire(parser.parse_args().destination)
