"""Tests for immutable scDFM source authentication."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import authenticate_scdfm_source as module  # noqa: E402


MIT_LICENSE = """MIT License

Copyright (c) 2025 Test Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the \"Software\"), to deal
in the Software without restriction.
"""


class AuthenticateScdfmSourceTests(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _repository(
        self,
        root: Path,
        *,
        omit: str | None = None,
    ) -> tuple[str, str, str]:
        for relative_name in module.REQUIRED_FILES:
            if relative_name == omit:
                continue
            path = root / relative_name
            path.parent.mkdir(parents=True, exist_ok=True)
            if relative_name == "LICENSE":
                path.write_text(MIT_LICENSE, encoding="utf-8")
            else:
                path.write_text(f"synthetic {relative_name}\n", encoding="utf-8")

        self._git(root, "init", "-q")
        self._git(root, "config", "user.email", "test@example.invalid")
        self._git(root, "config", "user.name", "scDFM Contract Test")
        self._git(root, "add", ".")
        self._git(root, "commit", "-q", "-m", "synthetic scDFM source")
        commit = self._git(root, "rev-parse", "HEAD^{commit}")
        tree = self._git(root, "rev-parse", "HEAD^{tree}")
        license_sha256 = hashlib.sha256(MIT_LICENSE.encode("utf-8")).hexdigest()
        return commit, tree, license_sha256

    def _authenticate(
        self,
        root: Path,
        commit: str,
        tree: str,
        license_sha256: str,
    ) -> dict[str, object]:
        return module.authenticate_source(
            root,
            expected_commit=commit,
            expected_tree=tree,
            expected_license_sha256=license_sha256,
        )

    def test_clean_pinned_checkout_passes_and_receipt_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scDFM"
            root.mkdir()
            commit, tree, license_sha256 = self._repository(root)

            first = self._authenticate(root, commit, tree, license_sha256)
            second = self._authenticate(root, commit, tree, license_sha256)

            self.assertEqual(first, second)
            self.assertEqual(first["schema"], module.SCHEMA)
            self.assertEqual(first["status"], "passed")
            self.assertEqual(first["source"]["repository_commit"], commit)
            self.assertEqual(first["source"]["repository_tree"], tree)
            self.assertEqual(
                first["source"]["license"]["sha256"], license_sha256
            )
            self.assertTrue(all(first["checks"].values()))

            output = Path(temporary) / "receipt.json"
            module.write_json_atomic(output, first)
            expected = json.dumps(first, indent=2, sort_keys=True) + "\n"
            self.assertEqual(output.read_text(encoding="utf-8"), expected)

    def test_wrong_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scDFM"
            root.mkdir()
            commit, tree, license_sha256 = self._repository(root)

            with self.assertRaisesRegex(module.AuthenticationError, "commit"):
                self._authenticate(root, "0" * 40, tree, license_sha256)

            self.assertNotEqual(commit, "0" * 40)

    def test_wrong_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scDFM"
            root.mkdir()
            commit, _, license_sha256 = self._repository(root)

            with self.assertRaisesRegex(module.AuthenticationError, "tree"):
                self._authenticate(root, commit, "0" * 40, license_sha256)

    def test_dirty_checkout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scDFM"
            root.mkdir()
            commit, tree, license_sha256 = self._repository(root)
            (root / "untracked.py").write_text("local change\n", encoding="utf-8")

            with self.assertRaisesRegex(module.AuthenticationError, "dirty"):
                self._authenticate(root, commit, tree, license_sha256)

    def test_missing_required_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "scDFM"
            root.mkdir()
            missing = "src/script/run.py"
            commit, tree, license_sha256 = self._repository(root, omit=missing)

            with self.assertRaisesRegex(
                module.AuthenticationError,
                f"Missing required scDFM file: {missing}",
            ):
                self._authenticate(root, commit, tree, license_sha256)

    def test_atomic_receipt_write_refuses_existing_path(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            output.write_text("original receipt\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic(output, {"status": "replacement"})

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "original receipt\n",
            )

    def test_concurrent_receipt_writers_publish_exactly_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "nested" / "receipt.json"
            writer_count = 8
            barrier = threading.Barrier(writer_count)

            def write(index: int) -> tuple[str, int]:
                barrier.wait()
                try:
                    module.write_json_atomic(output, {"writer": index})
                except FileExistsError:
                    return "exists", index
                return "written", index

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=writer_count
            ) as executor:
                results = list(executor.map(write, range(writer_count)))

            winners = [index for status, index in results if status == "written"]
            self.assertEqual(len(winners), 1)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"writer": winners[0]},
            )
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )


if __name__ == "__main__":
    unittest.main()
