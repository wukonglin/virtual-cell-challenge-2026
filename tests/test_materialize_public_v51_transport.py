"""Tests for immutable V5.1 sidecar transport materialization."""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_public_v51_transport as module  # noqa: E402


class PublicV51TransportTests(unittest.TestCase):
    def _fixture(self, root: Path) -> dict[str, Path | str]:
        panel_csv = root / "split_manifest.csv"
        panel_csv.write_bytes(b"target_gene,validation_route\nT1,direct\n")
        residual_npz = root / "hepg2_residual.npz"
        residual_npz.write_bytes(b"synthetic immutable residual payload")

        panel_source = root / "split_manifest.source.json"
        panel_source.write_text(
            json.dumps(
                {
                    "schema": module.PANEL_SCHEMA,
                    "selection_contract": {"effect_values_accessed": False},
                    "provenance": {
                        "output_csv": {
                            "path": "/cornell/original/split_manifest.csv",
                            "size_bytes": panel_csv.stat().st_size,
                            "sha256": module.sha256_file(panel_csv),
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        residual_source = root / "residual.source.json"
        residual_source.write_text(
            json.dumps(
                {
                    "schema": module.RESIDUAL_SCHEMA,
                    "context": "HepG2",
                    "contract": {"recipient_treated_profiles_used": False},
                    "provenance": {
                        "output_npz": {
                            "path": "/cornell/original/hepg2_residual.npz",
                            "size_bytes": residual_npz.stat().st_size,
                            "sha256": module.sha256_file(residual_npz),
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return {
            "panel_csv": panel_csv,
            "residual_npz": residual_npz,
            "panel_source": panel_source,
            "residual_source": residual_source,
            "panel_source_sha": module.sha256_file(panel_source),
            "residual_source_sha": module.sha256_file(residual_source),
            "panel_output": root / "transport" / "split_manifest.json",
            "residual_output": root / "transport" / "hepg2_residual.json",
        }

    def _argv(self, paths: dict[str, Path | str]) -> list[str]:
        return [
            "--panel-json-source",
            str(paths["panel_source"]),
            "--panel-json-expected-sha256",
            str(paths["panel_source_sha"]),
            "--panel-csv",
            str(paths["panel_csv"]),
            "--residual-json-source",
            str(paths["residual_source"]),
            "--residual-json-expected-sha256",
            str(paths["residual_source_sha"]),
            "--residual-npz",
            str(paths["residual_npz"]),
            "--output-panel-json",
            str(paths["panel_output"]),
            "--output-residual-json",
            str(paths["residual_output"]),
        ]

    def test_rebases_only_authenticated_paths_and_records_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            source_panel_bytes = Path(paths["panel_source"]).read_bytes()
            source_residual_bytes = Path(paths["residual_source"]).read_bytes()

            with contextlib.redirect_stdout(io.StringIO()):
                module.main(self._argv(paths))

            panel = json.loads(Path(paths["panel_output"]).read_text())
            residual = json.loads(Path(paths["residual_output"]).read_text())
            self.assertEqual(panel["schema"], module.PANEL_SCHEMA)
            self.assertEqual(residual["schema"], module.RESIDUAL_SCHEMA)
            self.assertEqual(
                panel["provenance"]["output_csv"]["path"],
                str(Path(paths["panel_csv"]).resolve()),
            )
            self.assertEqual(
                residual["provenance"]["output_npz"]["path"],
                str(Path(paths["residual_npz"]).resolve()),
            )
            for sidecar, key, payload in (
                (panel, "output_csv", Path(paths["panel_csv"])),
                (residual, "output_npz", Path(paths["residual_npz"])),
            ):
                descriptor = sidecar["provenance"][key]
                self.assertEqual(descriptor["size_bytes"], payload.stat().st_size)
                self.assertEqual(descriptor["sha256"], module.sha256_file(payload))
                transport = sidecar["transport_provenance"]
                self.assertEqual(transport["schema"], module.TRANSPORT_SCHEMA)
                self.assertTrue(transport["payload_bytes_preserved"])
                self.assertFalse(transport["scientific_content_changed"])
            self.assertEqual(Path(paths["panel_source"]).read_bytes(), source_panel_bytes)
            self.assertEqual(
                Path(paths["residual_source"]).read_bytes(), source_residual_bytes
            )

    def test_rejects_payload_mismatch_before_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            Path(paths["panel_csv"]).write_bytes(b"transport damage")

            with self.assertRaisesRegex(RuntimeError, "payload .* mismatch"):
                module.main(self._argv(paths))
            self.assertFalse(Path(paths["panel_output"]).exists())
            self.assertFalse(Path(paths["residual_output"]).exists())

    def test_rejects_source_sidecar_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            paths["panel_source_sha"] = "0" * 64

            with self.assertRaisesRegex(RuntimeError, "sidecar SHA-256 mismatch"):
                module.main(self._argv(paths))
            self.assertFalse(Path(paths["panel_output"]).exists())
            self.assertFalse(Path(paths["residual_output"]).exists())

    def test_refuses_output_overwrite_before_publishing_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            paths = self._fixture(Path(temporary))
            panel_output = Path(paths["panel_output"])
            panel_output.parent.mkdir(parents=True)
            panel_output.write_text("sentinel", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Refusing to overwrite"):
                module.main(self._argv(paths))
            self.assertEqual(panel_output.read_text(encoding="utf-8"), "sentinel")
            self.assertFalse(Path(paths["residual_output"]).exists())


if __name__ == "__main__":
    unittest.main()
