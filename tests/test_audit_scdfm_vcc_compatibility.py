"""Tests for the scDFM-to-VCC compatibility audit."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import audit_scdfm_vcc_compatibility as module  # noqa: E402


class AuditScdfmVccCompatibilityTests(unittest.TestCase):
    def _csv(self, path: Path, column: str, values: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([column])
            writer.writerows([[value] for value in values])

    def _selection(self, path: Path, weight: float = 1.0) -> str:
        payload = {
            "status": "passed",
            "decision": {
                "selected_arm": "g100",
                "state_effect_weight": weight,
            },
        }
        encoded = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
        path.write_bytes(encoded)
        return hashlib.sha256(encoded).hexdigest()

    def _fixture(self, root: Path) -> dict[str, object]:
        targets = root / "targets.csv"
        genes = root / "genes.csv"
        vocab = root / "vocab.json"
        selection = root / "selection.json"
        config = root / "config.toml"
        self._csv(targets, "target_gene", ["A", "B", "C"])
        self._csv(genes, "gene_name", ["A", "B", "C", "D", "E"])
        vocab.write_text(
            json.dumps({"<pad>": 0, "A": 1, "C": 2, "D": 3}),
            encoding="utf-8",
        )
        digest = self._selection(selection)
        config.write_text(
            f'''\
[experiment]
schema = "vcc-public-v7-scdfm-experiment-v1"
official_submission_allowed = false

[anchor]
selected_arm = "g100"
state_effect_weight = 1.0
immutable_during_scdfm_search = true
selection_receipt = "{selection.as_posix()}"
selection_receipt_sha256 = "{digest}"

[challenge]
measured_challenge_perturbation_data_allowed = false

[model]
role = "reliability_gated_residual_distribution"
absolute_scdfm_output_allowed = false
center_residual_by_context_target = true
''',
            encoding="utf-8",
        )
        return {
            "source": root,
            "targets_path": targets,
            "gene_axis_path": genes,
            "vocab_path": vocab,
            "p4_selection_path": selection,
            "expected_p4_sha256": digest,
            "config_path": config,
            "expected_target_count": 3,
            "expected_gene_count": 5,
            "source_authenticator": lambda _: {
                "schema": "test-source",
                "status": "passed",
            },
        }

    def test_valid_audit_is_deterministic_and_rejects_direct_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            first = module.audit_compatibility(**fixture)
            second = module.audit_compatibility(**fixture)

            self.assertEqual(first, second)
            self.assertEqual(first["status"], "passed")
            self.assertEqual(
                first["schema"], "vcc-scdfm-compatibility-audit-v2"
            )
            self.assertTrue(
                first["receipt_contract"]["adapter_config_sha256_required"]
            )
            adapter = first["authenticated_adapter_boundary"]
            self.assertTrue(adapter["config_sha256_bound"])
            self.assertEqual(
                first["inputs"]["adapter_config"]["sha256"],
                hashlib.sha256(fixture["config_path"].read_bytes()).hexdigest(),
            )
            target = first["coverage"]["vcc_targets_in_norman_vocab"]
            genes = first["coverage"]["vcc_gene_axis_in_norman_vocab"]
            self.assertEqual(target["covered"], ["A", "C"])
            self.assertEqual(target["missing"], ["B"])
            self.assertEqual(genes["covered"], ["A", "C", "D"])
            self.assertFalse(first["direct_checkpoint_eligibility"]["eligible"])
            self.assertEqual(first["anchor"]["state_effect_weight"], 1.0)
            firewall = first["data_firewall"]
            self.assertTrue(firewall["reads_only_axes_vocab_source_selection_and_config"])
            self.assertFalse(firewall["reads_challenge_control_expression"])
            self.assertFalse(firewall["reads_challenge_treated_expression"])
            self.assertFalse(firewall["reads_public_treated_expression"])
            self.assertFalse(firewall["reads_prediction_expression"])

    def test_non_gamma_one_anchor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            digest = self._selection(fixture["p4_selection_path"], weight=0.75)
            fixture["expected_p4_sha256"] = digest
            with self.assertRaisesRegex(module.CompatibilityError, "exactly 1.0"):
                module.audit_compatibility(**fixture)

    def test_wrong_selection_digest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            fixture["expected_p4_sha256"] = "0" * 64
            with self.assertRaisesRegex(module.CompatibilityError, "SHA-256"):
                module.audit_compatibility(**fixture)

    def test_permissive_adapter_config_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            config = fixture["config_path"]
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace(
                    "absolute_scdfm_output_allowed = false",
                    "absolute_scdfm_output_allowed = true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(module.CompatibilityError, "absolute"):
                module.audit_compatibility(**fixture)

    def test_legacy_config_without_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            config = fixture["config_path"]
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace(
                    'schema = "vcc-public-v7-scdfm-experiment-v1"\n',
                    "",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(module.CompatibilityError, "schema"):
                module.audit_compatibility(**fixture)

    def test_duplicate_target_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            self._csv(fixture["targets_path"], "target_gene", ["A", "A", "C"])
            with self.assertRaisesRegex(module.CompatibilityError, "Duplicate"):
                module.audit_compatibility(**fixture)

    def test_duplicate_vocab_index_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            fixture["vocab_path"].write_text(
                json.dumps({"<pad>": 0, "A": 1, "C": 1}), encoding="utf-8"
            )
            with self.assertRaisesRegex(module.CompatibilityError, "not unique"):
                module.audit_compatibility(**fixture)


if __name__ == "__main__":
    unittest.main()
