"""Adversarial tests for Arc ESM2 target-feature artifact authentication."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import stat
import sys
import tempfile
import threading
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import authenticate_esm2_target_features as module  # noqa: E402


class AuthenticateEsm2TargetFeaturesTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        features: dict[object, object] | None = None,
        targets: tuple[str, ...] = ("A", "C"),
    ) -> tuple[Path, Path, Path, module.TargetFeatureSpec]:
        root.mkdir(parents=True, exist_ok=True)
        if features is None:
            features = {
                "A": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "B": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                "C": torch.tensor([3.0, 4.0, 5.0, 6.0]),
            }
        feature_path = root / "features.pt"
        torch.save(features, feature_path)

        archive_path = root / "support.zip"
        member_name = "support/ESM2_pert_features.pt"
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            archive.writestr("support/README.txt", b"fixture")
            archive.write(feature_path, member_name)
        with zipfile.ZipFile(archive_path) as archive:
            member = archive.getinfo(member_name)

        target_path = root / "targets.csv"
        with target_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["target_gene"])
            writer.writerows([[target] for target in targets])

        tensor_features = features
        if not (
            type(tensor_features) is dict
            and tensor_features
            and all(isinstance(key, str) for key in tensor_features)
            and all(type(value) is torch.Tensor for value in tensor_features.values())
        ):
            # Invalid-payload tests need a syntactically valid spec.  The
            # canonical fields are deliberately borrowed from a valid map;
            # validation must reject the payload before comparing them.
            canonical_features = {
                "A": torch.tensor([1.0, 2.0, 3.0, 4.0]),
                "B": torch.tensor([2.0, 3.0, 4.0, 5.0]),
                "C": torch.tensor([3.0, 4.0, 5.0, 6.0]),
            }
        else:
            canonical_features = tensor_features

        ordered_target_digest = hashlib.sha256(
            json.dumps(list(targets), separators=(",", ":")).encode()
        ).hexdigest()
        subset_features = canonical_features
        subset_digest = hashlib.sha256()
        if all(target in subset_features for target in targets):
            subset_digest_value = module._target_subset_sha256(
                subset_features, targets
            )
        else:
            subset_digest_value = "0" * 64

        spec = module.TargetFeatureSpec(
            source_url="https://example.invalid/support.zip",
            source_object_provider="Google Cloud Storage",
            source_object_generation="1",
            source_object_last_modified="Thu, 01 Jan 1970 00:00:00 GMT",
            source_object_size_bytes=archive_path.stat().st_size,
            source_object_crc32c_base64="AAAAAA==",
            source_object_crc32c_hex="00000000",
            source_object_component_count=1,
            archive_filename=archive_path.name,
            archive_size_bytes=archive_path.stat().st_size,
            archive_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            archive_member=member_name,
            member_size_bytes=member.file_size,
            member_sha256=hashlib.sha256(feature_path.read_bytes()).hexdigest(),
            member_crc32=f"{member.CRC:08x}",
            extracted_filename=feature_path.name,
            feature_count=len(features),
            embedding_dimension=4,
            key_list_sha256=module._canonical_key_list_sha256(
                list(canonical_features)
            ),
            canonical_map_sha256=module._canonical_map_sha256(canonical_features),
            target_manifest_filename=target_path.name,
            target_manifest_size_bytes=target_path.stat().st_size,
            target_manifest_sha256=hashlib.sha256(
                target_path.read_bytes()
            ).hexdigest(),
            target_count=len(targets),
            ordered_target_list_sha256=ordered_target_digest,
            target_feature_subset_sha256=subset_digest_value,
        )
        return archive_path, feature_path, target_path, spec

    def test_valid_artifact_receipt_is_deterministic_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary) / "private" / "inputs")
            archive, features, targets, spec = paths
            first = module.authenticate_target_features(
                archive, features, targets, spec=spec
            )
            second = module.authenticate_target_features(
                archive, features, targets, spec=spec
            )

            self.assertEqual(first, second)
            self.assertEqual(first["schema"], module.SCHEMA)
            self.assertEqual(first["status"], "passed")
            self.assertTrue(all(first["checks"].values()))
            self.assertEqual(first["archive"]["sha256"], spec.archive_sha256)
            self.assertEqual(
                first["archive"]["member"]["sha256"], spec.member_sha256
            )
            self.assertEqual(
                first["feature_artifact"]["sha256"], spec.member_sha256
            )
            self.assertEqual(
                first["feature_artifact"]["post_load_sha256"], spec.member_sha256
            )
            self.assertEqual(
                first["feature_artifact"]["structure"]["feature_count"], 3
            )
            self.assertEqual(
                first["vcc_target_coverage"]["covered_targets"], 2
            )
            self.assertFalse(first["contract"]["provenance_complete"])
            self.assertFalse(first["contract"]["gate_1b_allowed"])
            self.assertTrue(first["contract"]["gate_1c_allowed"])
            self.assertTrue(
                first["contract"]["official_opaque_artifact_training_allowed"]
            )
            self.assertEqual(
                first["contract"]["opaque_artifact_training_mode"],
                "official_opaque_artifact",
            )
            self.assertEqual(
                first["archive"]["source_object"]["generation"],
                spec.source_object_generation,
            )
            self.assertFalse(
                first["archive"]["source_object"][
                    "source_url_claimed_permanently_immutable"
                ]
            )
            self.assertTrue(
                first["contract"]["does_not_establish_upstream_model_provenance"]
            )
            self.assertEqual(
                set(first["upstream_provenance"].values()),
                {"UNKNOWN_NOT_PROVIDED", False},
            )
            serialized = json.dumps(first, sort_keys=True)
            self.assertNotIn(str(Path(temporary)), serialized)

    def test_source_object_registration_is_internally_consistent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            cases = (
                (replace(spec, source_object_generation="mutable"), "generation"),
                (
                    replace(spec, source_object_size_bytes=spec.archive_size_bytes + 1),
                    "source-object size",
                ),
                (
                    replace(spec, source_object_crc32c_hex="ffffffff"),
                    "base64 and hex values disagree",
                ),
            )
            for invalid_spec, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    module.TargetFeatureAuthenticationError,
                    message,
                ):
                    module.authenticate_target_features(
                        archive,
                        features,
                        targets,
                        spec=invalid_spec,
                    )

    def test_reordered_mapping_has_the_same_canonical_hash(self) -> None:
        forward = {
            "A": torch.tensor([1.0, 2.0, 3.0, 4.0]),
            "B": torch.tensor([2.0, 3.0, 4.0, 5.0]),
        }
        reverse = dict(reversed(list(forward.items())))
        self.assertEqual(
            module._canonical_map_sha256(forward),
            module._canonical_map_sha256(reverse),
        )
        changed = dict(forward)
        changed["A"] = changed["A"].clone()
        changed["A"][0] += 1
        self.assertNotEqual(
            module._canonical_map_sha256(forward),
            module._canonical_map_sha256(changed),
        )

    def test_wrong_archive_sha_is_rejected_before_torch_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            with mock.patch.object(module.torch, "load") as loader:
                with self.assertRaisesRegex(
                    module.TargetFeatureAuthenticationError,
                    "support archive SHA-256",
                ):
                    module.authenticate_target_features(
                        archive,
                        features,
                        targets,
                        spec=replace(spec, archive_sha256="0" * 64),
                    )
                loader.assert_not_called()

    def test_member_sha_and_same_size_extracted_substitution_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError, "feature member SHA-256"
            ):
                module.authenticate_target_features(
                    archive,
                    features,
                    targets,
                    spec=replace(spec, member_sha256="0" * 64),
                )

            original = features.read_bytes()
            replacement = bytearray(original)
            replacement[-1] ^= 1
            features.write_bytes(replacement)
            self.assertEqual(features.stat().st_size, spec.member_size_bytes)
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "extracted feature artifact SHA-256",
            ):
                module.authenticate_target_features(
                    archive, features, targets, spec=spec
                )

    def test_weights_only_uses_the_authenticated_open_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            original_load = module.torch.load
            observed: dict[str, object] = {}

            def wrapped(handle: object, *args: object, **kwargs: object) -> object:
                observed["is_handle"] = hasattr(handle, "fileno")
                observed["map_location"] = kwargs.get("map_location")
                observed["weights_only"] = kwargs.get("weights_only")
                return original_load(handle, *args, **kwargs)

            with mock.patch.object(module.torch, "load", side_effect=wrapped):
                module.authenticate_target_features(
                    archive, features, targets, spec=spec
                )
            self.assertEqual(
                observed,
                {
                    "is_handle": True,
                    "map_location": "cpu",
                    "weights_only": True,
                },
            )

    def test_in_place_mutation_during_load_is_rejected_by_second_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            original_load = module.torch.load

            def mutate_after_load(
                handle: object, *args: object, **kwargs: object
            ) -> object:
                payload = original_load(handle, *args, **kwargs)
                with features.open("r+b") as writer:
                    writer.seek(-1, os.SEEK_END)
                    byte = writer.read(1)
                    writer.seek(-1, os.SEEK_END)
                    writer.write(bytes([byte[0] ^ 1]))
                    writer.flush()
                    os.fsync(writer.fileno())
                return payload

            with mock.patch.object(
                module.torch, "load", side_effect=mutate_after_load
            ), self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "bytes changed during restricted loading",
            ):
                module.authenticate_target_features(
                    archive, features, targets, spec=spec
                )

    def test_invalid_payload_types_shapes_values_and_target_coverage_fail(self) -> None:
        cases: tuple[tuple[str, dict[object, object], str], ...] = (
            (
                "non-string-key",
                {1: torch.ones(4), "B": torch.ones(4), "C": torch.ones(4)},
                "feature key",
            ),
            (
                "non-tensor",
                {"A": [1.0] * 4, "B": torch.ones(4), "C": torch.ones(4)},
                "plain tensor",
            ),
            (
                "wrong-dtype",
                {
                    "A": torch.ones(4, dtype=torch.float64),
                    "B": torch.ones(4),
                    "C": torch.ones(4),
                },
                "float32",
            ),
            (
                "wrong-shape",
                {"A": torch.ones(2, 2), "B": torch.ones(4), "C": torch.ones(4)},
                "feature shape",
            ),
            (
                "non-finite",
                {
                    "A": torch.tensor([1.0, 2.0, 3.0, float("nan")]),
                    "B": torch.ones(4),
                    "C": torch.ones(4),
                },
                "Non-finite",
            ),
            (
                "all-zero",
                {"A": torch.zeros(4), "B": torch.ones(4), "C": torch.ones(4)},
                "All-zero",
            ),
        )
        for label, payload, message in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                archive, features, targets, spec = self._fixture(
                    Path(temporary), features=payload
                )
                with self.assertRaisesRegex(
                    module.TargetFeatureAuthenticationError, message
                ):
                    module.authenticate_target_features(
                        archive, features, targets, spec=spec
                    )

        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(
                Path(temporary), targets=("A", "MISSING")
            )
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "missing requested VCC targets",
            ):
                module.authenticate_target_features(
                    archive, features, targets, spec=spec
                )

    def test_target_order_and_identity_are_cryptographically_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            archive, features, targets, spec = self._fixture(Path(temporary))
            raw = targets.read_text(encoding="utf-8")
            targets.write_text(raw.replace("A\nC", "C\nA"), encoding="utf-8")
            replacement_spec = replace(
                spec,
                target_manifest_sha256=hashlib.sha256(targets.read_bytes()).hexdigest(),
            )
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "ordered target-list SHA-256",
            ):
                module.authenticate_target_features(
                    archive, features, targets, spec=replacement_spec
                )

    def test_final_and_ancestor_symlinks_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive, features, targets, spec = self._fixture(root / "real")
            feature_link = root / spec.extracted_filename
            feature_link.symlink_to(features)
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "Unable to open extracted feature artifact",
            ):
                module.authenticate_target_features(
                    archive, feature_link, targets, spec=spec
                )

            ancestor = root / "linked-parent"
            ancestor.symlink_to(root / "real", target_is_directory=True)
            with self.assertRaisesRegex(
                module.TargetFeatureAuthenticationError,
                "Unable to open support archive",
            ):
                module.authenticate_target_features(
                    ancestor / archive.name,
                    features,
                    targets,
                    spec=spec,
                )

    def test_atomic_receipt_is_no_overwrite_and_concurrency_safe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipts" / "receipt.json"
            payload = {"schema": module.SCHEMA, "status": "passed"}
            barrier = threading.Barrier(8)
            outcomes: list[str] = []
            lock = threading.Lock()

            def writer() -> None:
                barrier.wait()
                try:
                    module.write_json_atomic(output, payload)
                    result = "written"
                except FileExistsError:
                    result = "exists"
                with lock:
                    outcomes.append(result)

            threads = [threading.Thread(target=writer) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()

            self.assertEqual(outcomes.count("written"), 1)
            self.assertEqual(outcomes.count("exists"), 7)
            self.assertEqual(json.loads(output.read_text()), payload)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic(output, {"status": "replacement"})
            self.assertEqual(json.loads(output.read_text()), payload)

    def test_atomic_receipt_rejects_symlinked_parent_without_external_write(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                module.write_json_atomic(alias / "receipt.json", {"status": "passed"})
            self.assertFalse((real / "receipt.json").exists())
            self.assertEqual(list(real.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
