"""Fast CPU tests for the synthetic V7 scDFM contract smoke."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
PROJECT = SCRIPTS.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smoke_scdfm_contract as module  # noqa: E402


class SmokeScdfmContractTests(unittest.TestCase):
    def _config(
        self,
        path: Path,
        *,
        state_effect_weight: float = 1.0,
        mmd_weight: float = 0.5,
        scales: str = "[0.5, 1.0, 2.0, 4.0]",
    ) -> Path:
        path.write_text(
            f"""\
[experiment]
schema = "test-scdfm-contract-v1"
name = "synthetic_contract_test"
seed = 12345
official_submission_allowed = false

[anchor]
state_effect_weight = {state_effect_weight!r}
immutable_during_scdfm_search = true

[model]
hidden_size = 16
flow_path = "linear_conditional_flow_matching"
mmd_weight = {mmd_weight!r}
mmd_kernel_scales = {scales}
absolute_scdfm_output_allowed = false
center_residual_by_context_target = true
""",
            encoding="utf-8",
        )
        return path

    def test_registered_v7_config_keeps_gamma_one_and_separate_mmd_weight(self) -> None:
        config = module.load_smoke_config(
            PROJECT / "configs/scdfm/vcc2026_v7_gamma1.toml"
        )
        self.assertEqual(config.state_effect_weight, 1.0)
        self.assertTrue(config.state_anchor_immutable)
        self.assertEqual(config.scdfm_mmd_weight, 0.5)
        self.assertEqual(config.mmd_kernel_scales, (0.5, 1.0, 2.0, 4.0))
        self.assertFalse(config.official_submission_allowed)
        self.assertFalse(config.absolute_scdfm_output_allowed)
        self.assertTrue(config.center_residual_by_context_target)

    def test_cpu_smoke_has_finite_cfm_mmd_and_backward_contract(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary) / "config.toml")
            launcher = Path(temporary) / "launcher.sbatch"
            launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            receipt = module.run_contract_smoke(
                config_path=config,
                requested_device="cpu",
                batch_size=4,
                synthetic_gene_count=8,
                launcher_path=launcher,
            )

        self.assertEqual(receipt["schema"], module.SCHEMA)
        self.assertEqual(receipt["status"], "passed")
        self.assertTrue(all(receipt["checks"].values()))
        self.assertEqual(receipt["device"]["type"], "cpu")
        self.assertEqual(
            receipt["training_contract"]["mmd_estimator"], module.MMD_ESTIMATOR
        )
        self.assertEqual(
            receipt["training_contract"]["mmd_kernel_scales"],
            [0.5, 1.0, 2.0, 4.0],
        )
        self.assertGreater(receipt["training_contract"]["gradient_l2_norm"], 0.0)
        self.assertFalse(receipt["scope"]["official_submission_artifact"])
        self.assertFalse(receipt["scope"]["model_performance_evaluation"])
        self.assertEqual(
            receipt["registered_files"]["launcher"]["filename"],
            "launcher.sbatch",
        )
        self.assertTrue(
            receipt["checks"][
                "registered_config_implementation_and_launcher_are_unchanged"
            ]
        )
        self.assertTrue(
            receipt["checks"][
                "configuration_was_parsed_from_authenticated_descriptor_bytes"
            ]
        )

    def test_run_does_not_reopen_config_through_path_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary) / "config.toml")
            with mock.patch.object(
                module,
                "load_smoke_config",
                side_effect=AssertionError("path loader must not be called"),
            ):
                receipt = module.run_contract_smoke(
                    config_path=config,
                    requested_device="cpu",
                    batch_size=4,
                    synthetic_gene_count=8,
                )

        self.assertEqual(receipt["status"], "passed")
        self.assertTrue(
            receipt["registered_files"]["configuration"][
                "bytes_read_and_hashed_from_same_descriptor"
            ]
        )

    def test_registered_file_descriptor_rejects_a_symlink_component(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            artifact = real / "config.toml"
            artifact.write_text("value = 1\n", encoding="utf-8")
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaisesRegex(module.ContractSmokeError, "Unable to open"):
                module.describe_regular_file(alias / "config.toml", "test file")

    def test_state_anchor_and_scdfm_mmd_weights_have_distinct_bindings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config_path = self._config(
                Path(temporary) / "config.toml", mmd_weight=0.25
            )
            receipt = module.run_contract_smoke(
                config_path=config_path,
                requested_device="cpu",
                batch_size=4,
                synthetic_gene_count=8,
            )

        state = receipt["weight_bindings"]["frozen_state_anchor_scale"]
        mmd = receipt["weight_bindings"]["scdfm_distribution_regularizer"]
        self.assertEqual(state["config_key"], "anchor.state_effect_weight")
        self.assertEqual(state["value"], 1.0)
        self.assertFalse(state["trainable"])
        self.assertEqual(mmd["config_key"], "model.mmd_weight")
        self.assertEqual(mmd["value"], 0.25)
        self.assertEqual(mmd["applies_only_to"], "multi_kernel_rbf_mmd2")
        expected = (
            receipt["losses"]["cfm_velocity_mse"]
            + receipt["losses"]["weighted_scdfm_mmd"]
        )
        self.assertAlmostEqual(receipt["losses"]["total"], expected, places=6)

    def test_non_gamma_one_state_anchor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(
                Path(temporary) / "config.toml", state_effect_weight=0.999
            )
            with self.assertRaisesRegex(
                module.ContractSmokeError, "must be exactly 1.0"
            ):
                module.load_smoke_config(config)

    def test_absolute_output_or_disabled_centering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            config = self._config(Path(temporary) / "config.toml")
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace(
                    "absolute_scdfm_output_allowed = false",
                    "absolute_scdfm_output_allowed = true",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(module.ContractSmokeError, "absolute"):
                module.load_smoke_config(config)

            config = self._config(Path(temporary) / "config2.toml")
            text = config.read_text(encoding="utf-8")
            config.write_text(
                text.replace(
                    "center_residual_by_context_target = true",
                    "center_residual_by_context_target = false",
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(module.ContractSmokeError, "center"):
                module.load_smoke_config(config)

    def test_multi_kernel_mmd_is_finite_and_backward_reaches_generated(self) -> None:
        generated = torch.tensor(
            [[0.0, 0.5], [1.0, -0.5], [0.25, 0.75]],
            dtype=torch.float32,
            requires_grad=True,
        )
        target = torch.tensor(
            [[0.1, 0.4], [0.8, -0.3], [0.4, 0.9]], dtype=torch.float32
        )
        value, median = module.multi_kernel_rbf_mmd2(
            generated, target, (0.5, 1.0, 2.0, 4.0)
        )
        value.backward()
        self.assertTrue(torch.isfinite(value).item())
        self.assertTrue(torch.isfinite(median).item())
        self.assertIsNotNone(generated.grad)
        self.assertTrue(torch.isfinite(generated.grad).all().item())

    def test_atomic_receipt_refuses_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            first = {"schema": "first", "value": 1}
            module.write_json_atomic_no_overwrite(output, first)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)
            self.assertFalse(
                any(path.suffix == ".tmp" for path in Path(temporary).iterdir())
            )

            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic_no_overwrite(
                    output, {"schema": "second", "value": 2}
                )
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), first)

    def test_receipt_writer_rejects_a_symlink_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                module.write_json_atomic_no_overwrite(
                    alias / "receipt.json",
                    {"schema": "must-not-write"},
                )
            self.assertFalse((real / "receipt.json").exists())


if __name__ == "__main__":
    unittest.main()
