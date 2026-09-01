"""CPU-only tests for the scDFM dependency preflight."""

from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import preflight_scdfm_environment as module  # noqa: E402

PINNED_SOURCE = Path(__file__).resolve().parents[1] / "dataset/external_repos/scDFM"


class EnvironmentPreflightTests(unittest.TestCase):
    @unittest.skipUnless(
        PINNED_SOURCE.is_dir(),
        "The Git-ignored pinned scDFM checkout is not installed",
    )
    def test_pinned_source_static_imports_are_fully_classified(self) -> None:
        discovered = module.discover_external_imports(PINNED_SOURCE)
        unknown = set(discovered) - set(module.REQUIRED_IMPORTS) - module.OPTIONAL_BACKENDS
        self.assertEqual(unknown, set())
        self.assertEqual(set(module.REQUIRED_IMPORTS) - set(discovered), set())
        self.assertIn("src/utils/_preprocessing.py", discovered["requests"])
        self.assertIn("src/flow_matching/ot/optimal_transport.py", discovered["ot"])

    def test_preflight_authenticates_before_importing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            (source / "entry.py").write_text("import mystery_package\n", encoding="utf-8")
            events: list[str] = []

            def authenticate(_: Path) -> dict[str, str]:
                events.append("authenticate")
                return {"status": "passed"}

            def importer(name: str) -> types.ModuleType:
                events.append(f"import:{name}")
                return types.ModuleType(name)

            with self.assertRaisesRegex(module.EnvironmentPreflightError, "Unmapped"):
                module.preflight_environment(
                    source,
                    source_authenticator=authenticate,
                    importer=importer,
                )
            self.assertEqual(events, ["authenticate"])

    @unittest.skipUnless(
        PINNED_SOURCE.is_dir(),
        "The Git-ignored pinned scDFM checkout is not installed",
    )
    def test_required_import_failure_is_actionable(self) -> None:
        source = PINNED_SOURCE

        def importer(name: str) -> types.ModuleType:
            if name == "rdkit":
                raise ModuleNotFoundError("rdkit")
            return types.ModuleType(name)

        with self.assertRaisesRegex(module.EnvironmentPreflightError, "rdkit"):
            module.preflight_environment(
                source,
                source_authenticator=lambda _: {"status": "passed"},
                importer=importer,
            )


if __name__ == "__main__":
    unittest.main()
