"""Tests for safe, immutable scDFM checkpoint auditing."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from collections import OrderedDict
from pathlib import Path
from unittest import mock

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_scdfm_checkpoint as module  # noqa: E402


def _unsafe_marker_write(path: str) -> dict[str, str]:
    Path(path).write_text("unsafe loader executed pickle global\n", encoding="utf-8")
    return {"executed": path}


class _UnsafePayload:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _unsafe_marker_write, (str(self.marker),)


class AuditScdfmCheckpointTests(unittest.TestCase):
    def _payload(self) -> dict[str, object]:
        return {
            "iteration": 17,
            "model_state_dict": OrderedDict(
                {
                    "perturbation_embedder.embedding.weight": torch.arange(
                        15, dtype=torch.float32
                    ).reshape(5, 3),
                    "decoder.weight": torch.ones(2, 3),
                }
            ),
            "optimizer_state_dict": {
                "state": {0: {"exp_avg": torch.zeros(2, 3)}},
                "param_groups": [{"lr": 1e-4, "params": [0]}],
            },
            "scheduler_state_dict": {"last_epoch": 16},
            "eval_score": 0.125,
            "config": {"model": "tiny"},
            "vocab": ["A", "B", "C", "D", "E"],
            "graph": {"edge_index": torch.tensor([[0, 1], [1, 2]])},
        }

    def _save(self, root: Path, payload: object | None = None) -> Path:
        checkpoint = root / "checkpoint.pt"
        torch.save(self._payload() if payload is None else payload, checkpoint)
        return checkpoint

    def test_valid_checkpoint_passes_and_records_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._save(root)
            expected_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

            receipt = module.audit_checkpoint(
                checkpoint,
                expected_sha256=expected_sha256.upper(),
            )

            self.assertEqual(receipt["schema"], module.SCHEMA)
            self.assertEqual(receipt["status"], "passed")
            self.assertEqual(receipt["checkpoint"]["sha256"], expected_sha256)
            self.assertEqual(receipt["structure"]["iteration"], 17)
            self.assertEqual(receipt["structure"]["eval_score"], 0.125)
            self.assertEqual(
                receipt["structure"]["perturbation_embedding"],
                {
                    "present": True,
                    "state_dict_key": "perturbation_embedder.embedding.weight",
                    "vocabulary_size": 5,
                    "hidden_dimension": 3,
                },
            )
            self.assertEqual(
                receipt["structure"]["tensors"]["checkpoint"]["tensor_count"],
                4,
            )
            self.assertEqual(
                receipt["structure"]["tensors"]["model_state_dict"]
                ["parameter_count_from_state_dict"],
                21,
            )
            for category in ("config", "vocab", "graph"):
                self.assertTrue(
                    receipt["structure"]["metadata"][category]["embedded"]
                )
            self.assertTrue(all(receipt["checks"].values()))

    def test_loader_is_restricted_to_cpu_weights_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._save(Path(temporary))
            original_load = torch.load
            calls: list[dict[str, object]] = []

            def observed_load(*args: object, **kwargs: object) -> object:
                calls.append(dict(kwargs))
                return original_load(*args, **kwargs)

            with mock.patch.object(module.torch, "load", side_effect=observed_load):
                module.audit_checkpoint(checkpoint)

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["map_location"], "cpu")
            self.assertIs(calls[0]["weights_only"], True)

    def test_wrong_hash_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._save(Path(temporary))

            with self.assertRaisesRegex(
                module.CheckpointAuditError, "SHA-256 mismatch"
            ):
                module.audit_checkpoint(checkpoint, expected_sha256="0" * 64)

    def test_missing_required_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload()
            del payload["scheduler_state_dict"]
            checkpoint = self._save(root, payload)

            with self.assertRaisesRegex(
                module.CheckpointAuditError,
                "missing required top-level keys: scheduler_state_dict",
            ):
                module.audit_checkpoint(checkpoint)

    def test_nonfinite_tensor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = self._payload()
            payload["optimizer_state_dict"]["state"][0]["exp_avg"][0, 0] = float(
                "nan"
            )
            checkpoint = self._save(root, payload)

            with self.assertRaisesRegex(
                module.CheckpointAuditError, "Non-finite tensor value"
            ):
                module.audit_checkpoint(checkpoint)

    def test_atomic_receipt_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._save(root)
            receipt = module.audit_checkpoint(checkpoint)
            output = root / "receipt.json"

            module.write_json_atomic(output, receipt)
            expected = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic(output, {"different": True})
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

    def test_symlink_and_nonregular_paths_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = self._save(root)
            link = root / "checkpoint-link.pt"
            link.symlink_to(checkpoint.name)

            with self.assertRaisesRegex(
                module.CheckpointAuditError, "must not be a symbolic link"
            ):
                module.audit_checkpoint(link)
            with self.assertRaisesRegex(
                module.CheckpointAuditError, "must be a regular file"
            ):
                module.audit_checkpoint(root)

    def test_restricted_loader_does_not_execute_pickle_globals(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "unsafe-marker.txt"
            checkpoint = self._save(root, {"payload": _UnsafePayload(marker)})

            with self.assertRaisesRegex(
                module.CheckpointAuditError, "Restricted weights-only"
            ):
                module.audit_checkpoint(checkpoint)
            self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
