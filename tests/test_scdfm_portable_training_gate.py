"""Adversarial tests for the repository-tracked V7 portable training gate."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scdfm_portable_training_gate import (  # noqa: E402
    PROTECTED_STAGES,
    PortableTrainingGateError,
    build_portable_hash_lock,
    canonical_sha256,
    formatted_json_bytes,
    load_portable_training_gate,
)
from train_scdfm_v7_portable import (  # noqa: E402
    _authenticate_code_file,
    _write_json_no_overwrite,
    run as run_portable_trainer,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PortableFixture:
    def __init__(
        self,
        root: Path,
        dataset: str = "nadig_jurkat",
        full_training: bool = False,
    ) -> None:
        self.root = root
        self.config = root / "configs/scdfm/vcc2026_v7_gamma1.toml"
        self.manifest = root / "artifacts/scdfm/v7/splits/jurkat_non_harm.json"
        self.preflight = (
            root
            / "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
        )
        self.lock = (
            root / "artifacts/scdfm/v7/training/portable_training_gate_lock.json"
        )
        source_name = (
            "synthetic_jurkat.h5ad" if dataset == "nadig_jurkat" else dataset
        )
        self.source = root / "dataset" / source_name
        for path in (
            self.config,
            self.manifest,
            self.preflight,
            self.lock,
            self.source,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        if full_training:
            import torch

            feature_path = (
                root
                / "dataset/state_support/extracted/ESM2_pert_features.pt"
            )
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"fit": torch.tensor([1.0, 2.0, 3.0, 4.0])},
                feature_path,
            )
            config_text = (
                "[experiment]\n"
                "seed = 20260901\n\n"
                "[model]\n"
                "hidden_size = 8\n"
                "mmd_weight = 0.5\n"
                "mmd_kernel_scales = [0.5, 1.0, 2.0]\n\n"
                "[target_conditioning_artifact_authentication]\n"
                'artifact = "dataset/state_support/extracted/'
                'ESM2_pert_features.pt"\n'
                f"artifact_size_bytes = {feature_path.stat().st_size}\n"
                f'artifact_sha256 = "{sha256(feature_path)}"\n'
                "embedding_dimension = 4\n"
            )
            self.labels = [
                "non-targeting",
                "non-targeting",
                "fit",
                "held",
                "fit",
            ]
        else:
            config_text = "[split]\nseed = 20260901\n"
            self.labels = ["non-targeting", "fit", "held", "fit"]
        self.config.write_text(config_text, encoding="utf-8")
        with h5py.File(self.source, "w") as handle:
            handle.create_dataset(
                "X",
                data=np.arange(
                    len(self.labels) * 6,
                    dtype=np.float32,
                ).reshape(len(self.labels), 6),
            )
            obs = handle.create_group("obs")
            column = "gene" if dataset == "nadig_jurkat" else "target_gene"
            obs.create_dataset(
                column,
                data=np.asarray(self.labels, dtype=h5py.string_dtype("utf-8")),
            )
        config_descriptor = {
            "filename": self.config.name,
            "size_bytes": self.config.stat().st_size,
            "sha256": sha256(self.config),
        }
        manifest_payload = {
            "schema": "vcc-scdfm-jurkat-non-harm-manifest-v1",
            "status": "sealed",
            "registration": {"configuration": config_descriptor},
            "partition": {
                "global_exclusion_targets": ["held"],
                "global_exclusion_target_count": 1,
                "global_exclusion_target_list_sha256": canonical_sha256(
                    ["held"]
                ),
            },
            "data_firewall": {
                "global_exclusion_scopes": list(PROTECTED_STAGES),
            },
        }
        self.manifest.write_bytes(formatted_json_bytes(manifest_payload))
        allowed = [
            index for index, label in enumerate(self.labels) if label != "held"
        ]
        excluded = [
            index for index, label in enumerate(self.labels) if label == "held"
        ]
        source_descriptor = {
            "filename": self.source.name,
            "size_bytes": self.source.stat().st_size,
            "sha256": sha256(self.source),
            "cell_count" if dataset == "nadig_jurkat" else "row_count": len(
                self.labels
            ),
        }
        preflight_payload = {
            "schema": "vcc-scdfm-jurkat-training-preflight-v1",
            "status": "passed",
            "configuration": config_descriptor,
            "sealed_manifest": {
                "filename": self.manifest.name,
                "size_bytes": self.manifest.stat().st_size,
                "sha256": sha256(self.manifest),
            },
            "global_exclusion_target_count": 1,
            "global_exclusion_target_list_sha256": canonical_sha256(["held"]),
            "global_exclusion_scopes": list(PROTECTED_STAGES),
            "datasets": [
                {
                    "dataset": dataset,
                    "source": source_descriptor,
                    "row_count": len(self.labels),
                    "allowed_row_count": len(allowed),
                    "excluded_row_count": len(excluded),
                    "allowed_source_row_indices_sha256": canonical_sha256(
                        allowed
                    ),
                    "excluded_source_row_indices_sha256": canonical_sha256(
                        excluded
                    ),
                    "allowed_target_list_sha256": canonical_sha256(
                        sorted(set(self.labels) - {"held"})
                    ),
                    "held_targets_absent_from_allowed_rows": True,
                }
            ],
            "consumer_contract": {
                "only_registered_allowed_source_row_indices_may_be_loaded": True,
                "complete_allowed_index_sequence_must_be_authorized_before_any_stage": True,
                "receipt_must_be_authenticated_before_model_fit": True,
                "receipt_must_be_authenticated_before_checkpoint_selection": True,
                "receipt_must_be_authenticated_before_normalization_or_feature_selection": True,
                "receipt_must_be_authenticated_before_graph_or_decoder_construction": True,
                "held_target_labels_in_any_allowed_batch_are_fatal": True,
                "expression_matrix_values_accessed_by_preflight": False,
                "production_training_ready": False,
                "portable_trainer_integration_implemented": False,
            },
        }
        self.preflight.write_bytes(formatted_json_bytes(preflight_payload))
        lock_payload = build_portable_hash_lock(self.manifest, self.preflight)
        self.lock.write_bytes(formatted_json_bytes(lock_payload))

    def load(self):
        return load_portable_training_gate(
            repository_root=self.root,
            lock_path=self.lock,
            manifest_path=self.manifest,
            preflight_path=self.preflight,
            split_configuration_path=self.config,
            expected_lock_sha256=sha256(self.lock),
        )


class ScdfmPortableTrainingGateTests(unittest.TestCase):
    def test_trainer_code_requires_independently_expected_hash(self) -> None:
        with self.assertRaisesRegex(
            PortableTrainingGateError,
            "independently expected SHA-256",
        ):
            _authenticate_code_file(
                SCRIPTS / "train_scdfm_v7_portable.py",
                "0" * 64,
                "portable trainer",
            )

    def test_lock_is_deterministic_and_location_independent(self) -> None:
        payloads = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as temporary:
                fixture = PortableFixture(Path(temporary))
                payloads.append(fixture.lock.read_bytes())
        self.assertEqual(payloads[0], payloads[1])

    def test_gate_authorizes_exact_rows_before_dense_batch_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            gate = fixture.load()
            self.assertEqual(gate.excluded_targets, frozenset({"held"}))
            with gate.open_h5_source("nadig_jurkat", fixture.source) as source:
                self.assertEqual(source.allowed_source_row_indices, (0, 1, 3))
                source.authorize_stage("normalization_fit")
                matrix = source.read_dense_expression(
                    (0, 1),
                    ("non-targeting", "fit"),
                    (0, 2, 4),
                    "normalization_fit",
                )
                self.assertEqual(matrix.shape, (2, 3))
                with self.assertRaisesRegex(
                    PortableTrainingGateError,
                    "labels do not match authenticated source rows",
                ):
                    source.validate_batch((0, 1), ("fit", "fit"), "normalization_fit")
                with self.assertRaisesRegex(
                    PortableTrainingGateError,
                    "unauthorized source rows",
                ):
                    source.validate_batch((1, 2), ("fit", "held"), "normalization_fit")
                with self.assertRaisesRegex(
                    PortableTrainingGateError,
                    "Gene index is negative",
                ):
                    source.read_dense_expression(
                        (0, 1),
                        ("non-targeting", "fit"),
                        (-1,),
                        "normalization_fit",
                    )

    def test_evaluation_only_source_cannot_be_promoted_by_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary), dataset="hepg2.h5")
            gate = fixture.load()
            with gate.open_h5_source("hepg2.h5", fixture.source) as source:
                with self.assertRaisesRegex(
                    PortableTrainingGateError,
                    "Evaluation-only source cannot enter model_fit",
                ):
                    source.authorize_stage("model_fit")

    def test_wrong_lock_hash_and_artifact_tampering_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            with self.assertRaisesRegex(
                PortableTrainingGateError,
                "independently expected SHA-256",
            ):
                load_portable_training_gate(
                    repository_root=fixture.root,
                    lock_path=fixture.lock,
                    manifest_path=fixture.manifest,
                    preflight_path=fixture.preflight,
                    split_configuration_path=fixture.config,
                    expected_lock_sha256="0" * 64,
                )
            fixture.preflight.write_bytes(fixture.preflight.read_bytes() + b" ")
            with self.assertRaises(PortableTrainingGateError):
                fixture.load()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            with h5py.File(fixture.source, "r+") as handle:
                handle["X"][0, 0] += 1.0
            gate = fixture.load()
            with self.assertRaisesRegex(
                PortableTrainingGateError,
                "Source SHA-256 changed",
            ):
                with gate.open_h5_source("nadig_jurkat", fixture.source):
                    pass

    def test_duplicate_keys_nonfinite_json_and_symlink_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            fixture.lock.write_text(
                '{"schema":"x","schema":"y"}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(PortableTrainingGateError, "Duplicate JSON key"):
                fixture.load()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            fixture.lock.write_text('{"value":NaN}\n', encoding="utf-8")
            with self.assertRaisesRegex(PortableTrainingGateError, "Non-finite JSON"):
                fixture.load()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            real = fixture.lock.with_name("real.json")
            fixture.lock.replace(real)
            fixture.lock.symlink_to(real.name)
            with self.assertRaisesRegex(PortableTrainingGateError, "Unable to open"):
                load_portable_training_gate(
                    repository_root=fixture.root,
                    lock_path=fixture.lock,
                    manifest_path=fixture.manifest,
                    preflight_path=fixture.preflight,
                    split_configuration_path=fixture.config,
                    expected_lock_sha256=sha256(real),
                )

    def test_portable_consumer_imports_no_split_builder_or_sklearn(self) -> None:
        code = (
            "import sys; "
            f"sys.path.insert(0, {str(SCRIPTS)!r}); "
            "import scdfm_portable_training_gate; "
            "assert 'scdfm_jurkat_non_harm' not in sys.modules; "
            "assert not any(name == 'sklearn' or name.startswith('sklearn.') "
            "for name in sys.modules)"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_trainer_preflight_consumes_gate_before_expression_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(Path(temporary))
            receipt = run_portable_trainer(
                Namespace(
                    repository_root=fixture.root,
                    lock=fixture.lock,
                    manifest=fixture.manifest,
                    preflight=fixture.preflight,
                    split_configuration=fixture.config,
                    expected_lock_sha256=sha256(fixture.lock),
                    expected_trainer_sha256=sha256(
                        SCRIPTS / "train_scdfm_v7_portable.py"
                    ),
                    expected_gate_consumer_sha256=sha256(
                        SCRIPTS / "scdfm_portable_training_gate.py"
                    ),
                    launcher=None,
                    expected_launcher_sha256=None,
                    expected_git_commit=None,
                    dataset="nadig_jurkat",
                    source=fixture.source,
                    output_json=fixture.root / "receipt.json",
                    device="cpu",
                    require_cuda=False,
                    require_h100=False,
                    batch_size=2,
                    gene_count=3,
                    optimizer_steps=1,
                    preflight_only=True,
                )
            )
            self.assertEqual(receipt["status"], "passed")
            self.assertFalse(receipt["authorization"]["expression_values_read"])
            self.assertIsNone(receipt["training"])
            self.assertTrue(
                receipt["checks"]
                ["complete_allowed_row_sequence_verified_before_fitted_stages"]
            )

    def test_trainer_full_cpu_smoke_authenticates_features_and_restarts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = PortableFixture(
                Path(temporary),
                full_training=True,
            )
            receipt = run_portable_trainer(
                Namespace(
                    repository_root=fixture.root,
                    lock=fixture.lock,
                    manifest=fixture.manifest,
                    preflight=fixture.preflight,
                    split_configuration=fixture.config,
                    expected_lock_sha256=sha256(fixture.lock),
                    expected_trainer_sha256=sha256(
                        SCRIPTS / "train_scdfm_v7_portable.py"
                    ),
                    expected_gate_consumer_sha256=sha256(
                        SCRIPTS / "scdfm_portable_training_gate.py"
                    ),
                    launcher=None,
                    expected_launcher_sha256=None,
                    expected_git_commit=None,
                    dataset="nadig_jurkat",
                    source=fixture.source,
                    output_json=fixture.root / "receipt.json",
                    device="cpu",
                    require_cuda=False,
                    require_h100=False,
                    batch_size=2,
                    gene_count=3,
                    optimizer_steps=1,
                    preflight_only=False,
                )
            )
            self.assertEqual(receipt["status"], "passed")
            self.assertTrue(receipt["authorization"]["expression_values_read"])
            self.assertEqual(
                receipt["target_features"]["embedding_dimension"],
                4,
            )
            self.assertTrue(
                receipt["training"]["deterministic_restart_identical"]
            )
            self.assertEqual(receipt["device"]["type"], "cpu")
            output = fixture.root / "receipt.json"
            _write_json_no_overwrite(output, receipt)
            self.assertTrue(output.is_file())
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                _write_json_no_overwrite(output, receipt)

    def test_repository_lock_exactly_rebuilds_from_tracked_artifacts(self) -> None:
        manifest = REPOSITORY / "artifacts/scdfm/v7/splits/jurkat_non_harm.json"
        preflight = (
            REPOSITORY
            / "artifacts/scdfm/v7/training/jurkat_training_preflight.json"
        )
        lock = (
            REPOSITORY
            / "artifacts/scdfm/v7/training/portable_training_gate_lock.json"
        )
        rebuilt = formatted_json_bytes(
            build_portable_hash_lock(manifest, preflight)
        )
        self.assertEqual(lock.read_bytes(), rebuilt)
        self.assertEqual(
            hashlib.sha256(rebuilt).hexdigest(),
            "f9c5b8675a1a46c561c056574009722d1faa4bfaf8b4e12e7d7017d3c5a9628d",
        )


if __name__ == "__main__":
    unittest.main()
