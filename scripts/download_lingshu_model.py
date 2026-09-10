#!/usr/bin/env python3
"""Download only the pinned public Lingshu model files on a CPU allocation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from extract_lingshu_gene_embeddings import (
    MODEL_FILES, MODEL_ID, MODEL_REVISION, authenticate_model_files, publish_bytes,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Model download must run inside a Slurm CPU or GPU allocation")
    from huggingface_hub import snapshot_download

    receipt = args.model_dir / "download_receipt.json"
    if os.path.lexists(receipt):
        authenticate_model_files(args.model_dir)
        previous = json.loads(receipt.read_text())
        if previous.get("model_id") != MODEL_ID or previous.get("model_revision") != MODEL_REVISION:
            raise ValueError("Existing download receipt has a different model identity")
        print(json.dumps({"status": "already_authenticated", "model_revision": MODEL_REVISION}))
        return 0
    snapshot_download(repo_id=MODEL_ID, revision=MODEL_REVISION, local_dir=args.model_dir,
                      allow_patterns=list(MODEL_FILES), token=False, max_workers=4)
    records = authenticate_model_files(args.model_dir)
    publish_bytes(receipt, (json.dumps({"schema": "vcc-lingshu-download-v1", "model_id": MODEL_ID,
                                       "model_revision": MODEL_REVISION, "files": records,
                                       "license": "MIT", "slurm_job_id": os.environ["SLURM_JOB_ID"]},
                                      sort_keys=True, indent=2) + "\n").encode())
    print(json.dumps({"status": "complete", "files": len(records), "model_revision": MODEL_REVISION}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
