"""Adversarial tests for the V7 centered continuous-factor contract."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import scdfm_centered_residual as module  # noqa: E402


class ScdfmCenteredResidualTests(unittest.TestCase):
    def _json(self, path: Path, payload: dict[str, object]) -> str:
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _axis(self, path: Path, column: str, values: list[str]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([column])
            writer.writerows([[value] for value in values])

    def _fixture(
        self,
        root: Path,
        *,
        official_submission_allowed: str = "false",
        absolute_output_allowed: str = "false",
        center_residual: str = "true",
        state_effect_weight: float = 1.0,
        centroid_atol: str = "1e-10",
        flow_semantics: str = "absolute_endpoint",
        contexts: tuple[str, ...] = ("A",),
    ) -> dict[str, object]:
        selection = root / "selection.json"
        selection_sha = self._json(
            selection,
            {
                "status": "passed",
                "decision": {"selected_arm": "g100", "state_effect_weight": 1.0},
            },
        )
        targets, genes = root / "targets.csv", root / "genes.csv"
        self._axis(targets, "target_gene", ["T1", "T2"])
        self._axis(genes, "gene_name", ["G1", "G2", "G3"])
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "contexts": list(contexts),
                    "n_genes": 3,
                    "n_constructs": 2,
                    "cells_per_pert": 4,
                    "per_context": {
                        context: {"n_perturbations": 2} for context in contexts
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        config = root / "config.toml"
        config.write_text(
            f"""\
[experiment]
schema = "test-v7"
seed = 123
official_submission_allowed = {official_submission_allowed}

[anchor]
selected_arm = "g100"
state_effect_weight = {state_effect_weight}
selection_receipt = "{selection.as_posix()}"
selection_receipt_sha256 = "{selection_sha}"
immutable_during_scdfm_search = true

[challenge]
targets = "{targets.as_posix()}"
gene_axis = "{genes.as_posix()}"
control_manifest = "{manifest.as_posix()}"
target_count = 2
gene_count = 3
cells_per_context_target = 4
measured_challenge_perturbation_data_allowed = false

[model]
absolute_scdfm_output_allowed = {absolute_output_allowed}
center_residual_by_context_target = {center_residual}
expression_space = "library_normalized_log1p"
centroid_atol = {centroid_atol}

[search]
scdfm_residual_weights = [0.0, 0.05, 0.10, 0.20]
""",
            encoding="utf-8",
        )
        groups = 2 * len(contexts)
        anchor = (1.0 + np.arange(groups * 4 * 3, dtype=np.float32)).reshape(groups, 4, 3)
        flow = anchor + np.flip(anchor, axis=1) * 0.05
        anchor_path, flow_path = root / "anchor.npy", root / "flow.npy"
        np.save(anchor_path, anchor, allow_pickle=False)
        np.save(flow_path, flow, allow_pickle=False)
        registration = module.read_registration(config)
        cell_axis_sha256 = hashlib.sha256(b"test-cell-axis-v1").hexdigest()
        latent_axis_sha256 = hashlib.sha256(b"test-latent-axis-v1").hexdigest()

        common = {
            "status": "passed",
            "measured_treated_data_used": False,
            "official_submission_artifact": False,
            "experiment_seed": 123,
            "representation": "library_normalized_log1p",
            "axes": {
                "target_axis_sha256": registration.target_axis.sha256,
                "gene_axis_sha256": registration.gene_axis.sha256,
                "context_manifest_sha256": registration.context_manifest.sha256,
                "canonical_group_layout_sha256": (
                    registration.canonical_group_layout_sha256
                ),
                "cell_axis_identity_sha256": cell_axis_sha256,
                "latent_axis_identity_sha256": latent_axis_sha256,
            },
        }
        anchor_receipt = root / "anchor_receipt.json"
        anchor_payload = {
            **common,
            "schema": module.ANCHOR_PROVENANCE_SCHEMA,
            "data_role": module.ANCHOR_DATA_ROLE,
            "artifact": module._identity(anchor_path, anchor, None),
            "p4_selection_sha256": selection_sha,
        }
        anchor_receipt_sha = self._json(anchor_receipt, anchor_payload)
        flow_receipt = root / "flow_receipt.json"
        flow_payload = {
            **common,
            "schema": module.FLOW_PROVENANCE_SCHEMA,
            "data_role": module.FLOW_DATA_ROLE,
            "artifact": module._identity(flow_path, flow, None),
            "flow_semantics": flow_semantics,
            "cell_axis_sha256": cell_axis_sha256,
            "latent_axis_sha256": latent_axis_sha256,
        }
        flow_receipt_sha = self._json(flow_receipt, flow_payload)
        return {
            "config": config,
            "selection": selection,
            "manifest": manifest,
            "anchor": anchor,
            "flow": flow,
            "anchor_path": anchor_path,
            "flow_path": flow_path,
            "anchor_receipt": anchor_receipt,
            "anchor_receipt_sha": anchor_receipt_sha,
            "flow_receipt": flow_receipt,
            "flow_receipt_sha": flow_receipt_sha,
            "flow_payload": flow_payload,
            "flow_semantics": flow_semantics,
        }

    def _execute_args(self, fixture: dict[str, object], root: Path, weight: float) -> dict[str, object]:
        return {
            "anchor_path": fixture["anchor_path"],
            "flow_path": fixture["flow_path"],
            "anchor_receipt_path": fixture["anchor_receipt"],
            "anchor_receipt_sha256": fixture["anchor_receipt_sha"],
            "flow_receipt_path": fixture["flow_receipt"],
            "flow_receipt_sha256": fixture["flow_receipt_sha"],
            "flow_semantics": fixture["flow_semantics"],
            "output_path": root / f"candidate_{weight}.npy",
            "receipt_path": root / f"candidate_{weight}.json",
            "config_path": fixture["config"],
            "weight": weight,
        }

    def test_residual_and_absolute_endpoint_semantics_are_distinct(self) -> None:
        anchor = 2.0 + np.arange(24, dtype=np.float64).reshape(2, 4, 3)
        endpoint = anchor + np.flip(anchor, axis=1) * 0.1
        residual = endpoint - anchor
        absolute_candidate, absolute_qc = module.build_centered_candidate(
            anchor,
            endpoint,
            weight=0.2,
            registered_weights=(0.0, 0.2),
            state_effect_weight=1.0,
            flow_semantics="absolute_endpoint",
        )
        residual_candidate, _ = module.build_centered_candidate(
            anchor,
            residual,
            weight=0.2,
            registered_weights=(0.0, 0.2),
            state_effect_weight=1.0,
            flow_semantics="residual_proposal",
        )
        np.testing.assert_allclose(absolute_candidate, residual_candidate, rtol=0, atol=1e-14)
        np.testing.assert_allclose(absolute_candidate.mean(axis=1), anchor.mean(axis=1), rtol=0, atol=1e-10)
        self.assertEqual(absolute_qc["flow_semantics"], "absolute_endpoint")

    def test_zero_arm_is_a_byte_identical_authenticated_anchor_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            args = self._execute_args(fixture, root, 0.0)
            receipt = module.execute(**args)
            output = args["output_path"]
            self.assertEqual(Path(output).read_bytes(), Path(fixture["anchor_path"]).read_bytes())
            self.assertEqual(receipt["output"]["file_sha256"], hashlib.sha256(Path(output).read_bytes()).hexdigest())
            self.assertTrue(receipt["diagnostics"]["zero_arm_byte_identical_to_anchor"])
            self.assertFalse(receipt["scope"]["raw_count_invariance_claimed"])
            self.assertFalse(receipt["scope"]["official_submission_artifact"])

    def test_nonzero_arm_is_continuous_only_and_preserves_centroid(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            args = self._execute_args(fixture, root, 0.1)
            receipt = module.execute(**args)
            candidate = np.load(args["output_path"], allow_pickle=False)
            np.testing.assert_allclose(candidate.mean(axis=1), fixture["anchor"].mean(axis=1), rtol=0, atol=1e-10)
            self.assertEqual(receipt["contract"]["flow_semantics"], "absolute_endpoint")
            self.assertTrue(receipt["scope"]["continuous_expression_factor_only"])
            self.assertTrue(receipt["scope"]["requires_authenticated_count_renderer"])

    def test_config_firewalls_and_p4_hash_are_enforced(self) -> None:
        cases = [
            ({"official_submission_allowed": "true"}, "official_submission_allowed"),
            ({"absolute_output_allowed": "true"}, "absolute_scdfm_output_allowed"),
            ({"center_residual": "false"}, "center_residual_by_context_target"),
            ({"state_effect_weight": 0.75}, "exactly 1.0"),
            ({"centroid_atol": "1e-3"}, "centroid_atol"),
        ]
        for overrides, message in cases:
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(module.ResidualContractError, message):
                    self._fixture(Path(temporary), **overrides)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            Path(fixture["selection"]).write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(module.ResidualContractError, "P4 selection SHA-256"):
                module.read_registration(Path(fixture["config"]))

    def test_adversarial_provenance_role_hash_seed_and_axes_are_rejected(self) -> None:
        mutations = [
            ("data_role", "measured_treated_expression", "data role"),
            ("measured_treated_data_used", True, "Measured treated"),
            ("experiment_seed", 999, "seed mismatch"),
        ]
        for field, value, message in mutations:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                payload = dict(fixture["flow_payload"])
                payload[field] = value
                fixture["flow_receipt_sha"] = self._json(Path(fixture["flow_receipt"]), payload)
                with self.assertRaisesRegex(module.ResidualContractError, message):
                    module.execute(**self._execute_args(fixture, root, 0.1))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            with self.assertRaisesRegex(module.ResidualContractError, "receipt SHA-256 mismatch"):
                args = self._execute_args(fixture, root, 0.1)
                args["flow_receipt_sha256"] = "0" * 64
                module.execute(**args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            payload = dict(fixture["flow_payload"])
            payload["axes"] = {
                "target_axis_sha256": "0" * 64,
                "gene_axis_sha256": payload["axes"]["gene_axis_sha256"],
            }
            fixture["flow_receipt_sha"] = self._json(Path(fixture["flow_receipt"]), payload)
            with self.assertRaisesRegex(module.ResidualContractError, "Target-axis"):
                module.execute(**self._execute_args(fixture, root, 0.1))

    def test_flow_semantics_must_match_authenticated_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, flow_semantics="residual_proposal")
            args = self._execute_args(fixture, root, 0.1)
            args["flow_semantics"] = "absolute_endpoint"
            with self.assertRaisesRegex(module.ResidualContractError, "Flow semantics"):
                module.execute(**args)

    def test_permuted_context_manifest_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, contexts=("A", "B"))
            manifest = Path(fixture["manifest"])
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            payload["contexts"] = ["B", "A"]
            # Keep the registered per_context insertion order A, B. A permutation
            # must not silently redefine the flattened group order.
            manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(module.ResidualContractError, "order"):
                module.execute(**self._execute_args(fixture, root, 0.1))

    def test_symlink_inputs_receipts_config_and_dangling_outputs_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            anchor_link = root / "anchor_link.npy"
            anchor_link.symlink_to(fixture["anchor_path"])
            args = self._execute_args(fixture, root, 0.1)
            args["anchor_path"] = anchor_link
            with self.assertRaisesRegex(module.ResidualContractError, "symbolic link"):
                module.execute(**args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            receipt_link = root / "flow_receipt_link.json"
            receipt_link.symlink_to(fixture["flow_receipt"])
            args = self._execute_args(fixture, root, 0.1)
            args["flow_receipt_path"] = receipt_link
            with self.assertRaisesRegex(module.ResidualContractError, "symbolic link"):
                module.execute(**args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            config_link = root / "config_link.toml"
            config_link.symlink_to(fixture["config"])
            args = self._execute_args(fixture, root, 0.1)
            args["config_path"] = config_link
            with self.assertRaisesRegex(module.ResidualContractError, "symbolic link"):
                module.execute(**args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            manifest = Path(fixture["manifest"])
            real_manifest = root / "manifest_real.json"
            manifest.rename(real_manifest)
            manifest.symlink_to(real_manifest)
            with self.assertRaisesRegex(module.ResidualContractError, "symbolic link"):
                module.execute(**self._execute_args(fixture, root, 0.1))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            args = self._execute_args(fixture, root, 0.1)
            dangling = Path(args["output_path"])
            dangling.symlink_to(root / "missing-target.npy")
            with self.assertRaisesRegex(FileExistsError, "symbolic link"):
                module.execute(**args)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            args = self._execute_args(fixture, root, 0.1)
            dangling = Path(args["receipt_path"])
            dangling.symlink_to(root / "missing-receipt.json")
            with self.assertRaisesRegex(FileExistsError, "symbolic link"):
                module.execute(**args)

    def test_missing_o_nofollow_support_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            with mock.patch.object(module.os, "O_NOFOLLOW", None):
                with self.assertRaisesRegex(
                    module.ResidualContractError, "no-follow file opening is unavailable"
                ):
                    module.sha256_file(Path(fixture["anchor_path"]))

    def test_cell_and_latent_axis_identity_mismatches_are_rejected(self) -> None:
        for field in ("cell_axis_identity_sha256", "latent_axis_identity_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                payload = dict(fixture["flow_payload"])
                payload["axes"] = dict(payload["axes"])
                payload["axes"][field] = hashlib.sha256(field.encode("utf-8")).hexdigest()
                fixture["flow_receipt_sha"] = self._json(
                    Path(fixture["flow_receipt"]), payload
                )
                with self.assertRaisesRegex(module.ResidualContractError, field):
                    module.execute(**self._execute_args(fixture, root, 0.1))

    def test_placeholder_cell_and_latent_axis_hashes_are_rejected(self) -> None:
        for field, message in (
            ("cell_axis_identity_sha256", "Cell-axis identity"),
            ("latent_axis_identity_sha256", "Latent-axis identity"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fixture = self._fixture(root)
                payload = dict(fixture["flow_payload"])
                payload["axes"] = dict(payload["axes"])
                payload["axes"][field] = "0" * 64
                fixture["flow_receipt_sha"] = self._json(
                    Path(fixture["flow_receipt"]), payload
                )
                with self.assertRaisesRegex(module.ResidualContractError, message):
                    module.execute(**self._execute_args(fixture, root, 0.1))

    def test_canonical_group_layout_hash_is_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            payload = dict(fixture["flow_payload"])
            payload["axes"] = dict(payload["axes"])
            payload["axes"]["canonical_group_layout_sha256"] = "0" * 64
            fixture["flow_receipt_sha"] = self._json(
                Path(fixture["flow_receipt"]), payload
            )
            with self.assertRaisesRegex(module.ResidualContractError, "group-layout"):
                module.execute(**self._execute_args(fixture, root, 0.1))

    def test_registered_shape_nonfinite_weight_and_state_are_enforced(self) -> None:
        valid = np.ones((2, 4, 3), dtype=np.float64)
        with self.assertRaisesRegex(module.ResidualContractError, "shapes differ"):
            module.build_centered_candidate(valid, valid[:, :, :2], weight=0.1, registered_weights=(0.1,), state_effect_weight=1.0)
        invalid = valid.copy(); invalid[0, 0, 0] = np.nan
        with self.assertRaisesRegex(module.ResidualContractError, "non-finite"):
            module.build_centered_candidate(valid, invalid, weight=0.1, registered_weights=(0.1,), state_effect_weight=1.0)
        with self.assertRaisesRegex(module.ResidualContractError, "not registered"):
            module.build_centered_candidate(valid, valid, weight=0.15, registered_weights=(0.1,), state_effect_weight=1.0)
        with self.assertRaisesRegex(module.ResidualContractError, "exactly 1.0"):
            module.build_centered_candidate(valid, valid, weight=0.1, registered_weights=(0.1,), state_effect_weight=0.9)

    def test_centroid_tolerance_is_not_a_cli_override(self) -> None:
        parsed = module.parse_args(
            [
                "--anchor", "a.npy",
                "--flow", "f.npy",
                "--anchor-receipt", "a.json",
                "--anchor-receipt-sha256", "0" * 64,
                "--flow-receipt", "f.json",
                "--flow-receipt-sha256", "1" * 64,
                "--flow-semantics", "absolute_endpoint",
                "--output", "o.npy",
                "--receipt", "o.json",
                "--weight", "0.1",
            ]
        )
        self.assertFalse(hasattr(parsed, "centroid_atol"))
        self.assertEqual(module.REGISTERED_CENTROID_ATOL, 1e-10)


if __name__ == "__main__":
    unittest.main()
