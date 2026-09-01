"""Tests for extraction-free scDFM asset authentication."""

from __future__ import annotations

import hashlib
import json
import stat
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import authenticate_scdfm_assets as module  # noqa: E402


class AuthenticateScdfmAssetsTests(unittest.TestCase):
    def _write_zip(
        self,
        path: Path,
        members: list[tuple[str | zipfile.ZipInfo, bytes]],
    ) -> None:
        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members:
                archive.writestr(name, content)

    def _fixture(
        self, root: Path
    ) -> tuple[dict[str, Path], tuple[module.AssetSpec, ...]]:
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        definitions = {
            "checkpoints": [
                ("ckpts/additive/fold0/checkpoint.pt", b"model-zero"),
                ("ckpts/holdout/fold0/checkpoint.pt", b"model-one"),
                ("ckpts/results.csv", b"metric,value\nmse,0.1\n"),
            ],
            "norman": [
                ("norman/norman.h5ad", b"synthetic h5ad"),
                ("norman/split_results.pkl", b"synthetic split"),
            ],
            "combosciplex": [
                ("combosciplex/combosciplex.h5ad", b"synthetic combo"),
            ],
        }
        specs = []
        for key, members in definitions.items():
            path = root / f"{key}.zip"
            self._write_zip(path, members)
            paths[key] = path
            specs.append(
                module.AssetSpec(
                    key=key,
                    official_filename=f"official-{key}.zip",
                    google_drive_file_id=f"drive-{key}",
                    expected_size_bytes=path.stat().st_size,
                )
            )
        return paths, tuple(specs)

    def test_three_archives_pass_and_receipt_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "private" / "download-cache"
            root.mkdir(parents=True)
            paths, specs = self._fixture(root)

            first = module.authenticate_assets(paths, specs=specs)
            second = module.authenticate_assets(paths, specs=specs)

            self.assertEqual(first, second)
            self.assertEqual(first["schema"], module.SCHEMA)
            self.assertEqual(
                first["schema"], "vcc-scdfm-assets-authentication-v2"
            )
            self.assertEqual(first["status"], "passed")
            self.assertFalse(
                first["contract"]["legacy_size_only_receipts_accepted"]
            )
            self.assertTrue(all(first["checks"].values()))
            self.assertEqual(
                first["assets"]["checkpoints"]["zip"]["checkpoint_count"], 2
            )
            self.assertEqual(
                first["assets"]["norman"]["zip"]["member_count"], 2
            )
            self.assertEqual(
                first["assets"]["combosciplex"]["zip"]["member_count"], 1
            )
            for key, path in paths.items():
                descriptor = first["assets"][key]
                self.assertEqual(descriptor["local_filename"], path.name)
                self.assertEqual(
                    descriptor["sha256"],
                    hashlib.sha256(path.read_bytes()).hexdigest(),
                )
                self.assertEqual(
                    descriptor["observed_size_bytes"], path.stat().st_size
                )
                summary = descriptor["zip"]
                self.assertGreater(summary["compressed_member_bytes"], 0)
                self.assertGreater(summary["uncompressed_member_bytes"], 0)

            serialized = json.dumps(first, sort_keys=True)
            self.assertNotIn(str(root), serialized)
            self.assertNotIn(str(Path(temporary)), serialized)

    def test_optional_extracted_files_match_zip_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive_root = root / "archives" / "private-cache"
            archive_root.mkdir(parents=True)
            paths, specs = self._fixture(archive_root)
            extracted_root = root / "extracted" / "private-root"
            extracted_contents = {
                "norman/norman.h5ad": b"synthetic h5ad",
                "norman/split_results.pkl": b"synthetic split",
                "combosciplex/combosciplex.h5ad": b"synthetic combo",
            }
            for relative_name, content in extracted_contents.items():
                target = extracted_root / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            receipt = module.authenticate_assets(
                paths,
                specs=specs,
                extracted_root=extracted_root,
            )

            self.assertEqual(
                set(receipt["extracted_files"]), set(extracted_contents)
            )
            for relative_name, content in extracted_contents.items():
                descriptor = receipt["extracted_files"][relative_name]
                self.assertEqual(descriptor["local_relative_path"], relative_name)
                self.assertEqual(
                    descriptor["expected_size_bytes_from_zip"], len(content)
                )
                self.assertEqual(descriptor["observed_size_bytes"], len(content))
                self.assertEqual(
                    descriptor["sha256"], hashlib.sha256(content).hexdigest()
                )
                self.assertEqual(
                    descriptor["expected_sha256_from_zip"],
                    hashlib.sha256(content).hexdigest(),
                )
                self.assertTrue(descriptor["sha256_matches_zip_member"])

            serialized = json.dumps(receipt, sort_keys=True)
            self.assertNotIn(str(extracted_root), serialized)
            self.assertNotIn(str(archive_root), serialized)

    def test_extracted_file_wrong_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, specs = self._fixture(root / "archives")
            extracted_root = root / "extracted"
            contents = {
                "norman/norman.h5ad": b"wrong",
                "norman/split_results.pkl": b"synthetic split",
                "combosciplex/combosciplex.h5ad": b"synthetic combo",
            }
            for relative_name, content in contents.items():
                target = extracted_root / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            with self.assertRaisesRegex(
                module.AssetAuthenticationError,
                "Unexpected extracted member size",
            ):
                module.authenticate_assets(
                    paths,
                    specs=specs,
                    extracted_root=extracted_root,
                )

    def test_same_size_extracted_substitution_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, specs = self._fixture(root / "archives")
            extracted_root = root / "extracted"
            contents = {
                # Changing only case preserves the archive member byte length.
                "norman/norman.h5ad": b"Synthetic h5ad",
                "norman/split_results.pkl": b"synthetic split",
                "combosciplex/combosciplex.h5ad": b"synthetic combo",
            }
            self.assertEqual(len(contents["norman/norman.h5ad"]), len(b"synthetic h5ad"))
            for relative_name, content in contents.items():
                target = extracted_root / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            with self.assertRaisesRegex(
                module.AssetAuthenticationError,
                "SHA-256 does not match ZIP member",
            ):
                module.authenticate_assets(
                    paths,
                    specs=specs,
                    extracted_root=extracted_root,
                )

    def test_extracted_symlink_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, specs = self._fixture(root / "archives")
            extracted_root = root / "extracted"
            (extracted_root / "norman").mkdir(parents=True)
            outside = root / "outside.h5ad"
            outside.write_bytes(b"synthetic h5ad")
            (extracted_root / "norman" / "norman.h5ad").symlink_to(outside)
            (extracted_root / "norman" / "split_results.pkl").write_bytes(
                b"synthetic split"
            )
            (extracted_root / "combosciplex").mkdir(parents=True)
            (
                extracted_root / "combosciplex" / "combosciplex.h5ad"
            ).write_bytes(b"synthetic combo")

            with self.assertRaisesRegex(
                module.AssetAuthenticationError,
                "contains a symlink",
            ):
                module.authenticate_assets(
                    paths,
                    specs=specs,
                    extracted_root=extracted_root,
                )

    def test_legacy_size_only_member_binding_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, specs = self._fixture(root / "archives")
            assets = module.authenticate_assets(paths, specs=specs)["assets"]
            binding = assets["norman"]["required_extracted_members"][
                "norman/norman.h5ad"
            ]
            binding.pop("expected_sha256")
            extracted_root = root / "extracted"
            for relative_name, content in {
                "norman/norman.h5ad": b"synthetic h5ad",
                "norman/split_results.pkl": b"synthetic split",
                "combosciplex/combosciplex.h5ad": b"synthetic combo",
            }.items():
                target = extracted_root / relative_name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)

            with self.assertRaisesRegex(
                module.AssetAuthenticationError,
                "Missing bound member SHA-256",
            ):
                module.authenticate_extracted_files(extracted_root, assets)

    def test_wrong_exact_byte_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths, specs = self._fixture(root)
            wrong_specs = tuple(
                module.AssetSpec(
                    key=spec.key,
                    official_filename=spec.official_filename,
                    google_drive_file_id=spec.google_drive_file_id,
                    expected_size_bytes=(
                        spec.expected_size_bytes + 1
                        if spec.key == "norman"
                        else spec.expected_size_bytes
                    ),
                )
                for spec in specs
            )

            with self.assertRaisesRegex(
                module.AssetAuthenticationError, "Unexpected norman archive size"
            ):
                module.authenticate_assets(paths, specs=wrong_specs)

    def test_non_zip_with_registered_size_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "not-a-zip.zip"
            path.write_bytes(b"not a zip archive")
            spec = module.AssetSpec("broken", "broken.zip", "drive-broken", 17)

            with self.assertRaisesRegex(
                module.AssetAuthenticationError, "structurally valid readable ZIP"
            ):
                module.authenticate_archive(path, spec)

    def test_wrong_registered_sha256_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "valid.zip"
            self._write_zip(path, [("data/payload.bin", b"payload")])
            spec = module.AssetSpec(
                "valid",
                "valid.zip",
                "drive-valid",
                path.stat().st_size,
                "0" * 64,
            )
            with self.assertRaisesRegex(
                module.AssetAuthenticationError, "archive SHA-256"
            ):
                module.authenticate_archive(path, spec)

    def test_crc_corruption_is_rejected_without_extraction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "corrupt.zip"
            self._write_zip(path, [("data/payload.bin", b"A" * 2048)])
            with zipfile.ZipFile(path) as archive:
                info = archive.getinfo("data/payload.bin")
                with path.open("r+b") as handle:
                    handle.seek(info.header_offset + 26)
                    name_length = int.from_bytes(handle.read(2), "little")
                    extra_length = int.from_bytes(handle.read(2), "little")
                    data_offset = info.header_offset + 30 + name_length + extra_length
                    handle.seek(data_offset)
                    original = handle.read(1)
                    handle.seek(data_offset)
                    handle.write(bytes([original[0] ^ 0xFF]))

            spec = module.AssetSpec(
                "corrupt", "corrupt.zip", "drive-corrupt", path.stat().st_size
            )
            with self.assertRaisesRegex(
                module.AssetAuthenticationError,
                "structurally valid readable ZIP|CRC validation",
            ):
                module.authenticate_archive(path, spec)

    def test_absolute_parent_and_windows_member_paths_are_rejected(self) -> None:
        unsafe_names = (
            "/absolute/file.txt",
            "../parent.txt",
            "safe/../../parent.txt",
            r"C:\private\file.txt",
            r"safe\..\parent.txt",
            r"\\server\share\file.txt",
        )
        for unsafe_name in unsafe_names:
            with self.subTest(name=unsafe_name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "unsafe.zip"
                self._write_zip(path, [(unsafe_name, b"unsafe")])
                spec = module.AssetSpec(
                    "unsafe", "unsafe.zip", "drive-unsafe", path.stat().st_size
                )
                with self.assertRaises(module.AssetAuthenticationError):
                    module.authenticate_archive(path, spec)

    def test_symbolic_link_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "symlink.zip"
            link = zipfile.ZipInfo("data/link")
            link.create_system = 3
            link.external_attr = (stat.S_IFLNK | 0o777) << 16
            self._write_zip(path, [(link, b"../outside")])
            spec = module.AssetSpec(
                "symlink", "symlink.zip", "drive-symlink", path.stat().st_size
            )

            with self.assertRaisesRegex(
                module.AssetAuthenticationError, "symbolic link"
            ):
                module.authenticate_archive(path, spec)

    def test_duplicate_normalized_member_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "duplicate.zip"
            self._write_zip(
                path,
                [("data/file.txt", b"first"), ("data//file.txt", b"second")],
            )
            spec = module.AssetSpec(
                "duplicate", "duplicate.zip", "drive-duplicate", path.stat().st_size
            )

            with self.assertRaisesRegex(
                module.AssetAuthenticationError, "duplicate normalized member"
            ):
                module.authenticate_archive(path, spec)

    def test_atomic_receipt_write_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "receipt.json"
            payload = {"schema": module.SCHEMA, "status": "passed"}
            module.write_json_atomic(output, payload)
            expected = json.dumps(payload, indent=2, sort_keys=True) + "\n"
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic(output, {"status": "replacement"})
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

    def test_asset_keys_must_exactly_match_specs(self) -> None:
        with self.assertRaisesRegex(
            module.AssetAuthenticationError, "keys must exactly match"
        ):
            module.authenticate_assets(
                {},
                specs=(module.AssetSpec("one", "one.zip", "drive-one", 1),),
            )


if __name__ == "__main__":
    unittest.main()
