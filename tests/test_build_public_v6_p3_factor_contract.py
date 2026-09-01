"""Focused tests for the immutable four-arm P3 factor contract."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_public_v6_p3_factor_contract as module  # noqa: E402


class PublicV6P3FactorContractTests(unittest.TestCase):
    @staticmethod
    def _generation(
        arm: str, prediction_descriptor: dict[str, object]
    ) -> dict[str, object]:
        configuration = {
            "context": "HepG2",
            "cells_per_group": 2,
            "seed": 20260901,
            "target_policy": "force",
            **module.FACTOR_LEVELS[arm],
        }
        return {
            "schema": module.PREDICTION_SCHEMA,
            "artifact_type": "sealed_public_validation_prediction",
            "full_frozen_panel_contract": True,
            "data_firewall": {
                "sealed_treated_profiles_read": False,
                "truth_inputs_read": [],
            },
            "configuration": configuration,
            "validation": {
                "shape": [8, 3],
                "groups": 4,
                "failed_checks": [],
                "checks": {"shape_exact": True},
            },
            "scientific_qc": {
                "all_libraries_exact": True,
                "all_native_non_support_counts_exact": True,
                "target_knockdown_failures": [],
            },
            "provenance": {"output_h5ad": prediction_descriptor},
        }

    def _arm(self, root: Path, arm: str, **configuration_changes: object) -> module.FactorArm:
        prediction = root / f"{arm}.h5ad"
        prediction.write_bytes(arm.encode("ascii"))
        prediction_descriptor = module.describe_file(prediction)
        generation = self._generation(arm, prediction_descriptor)
        generation["configuration"].update(configuration_changes)
        generation_path = root / f"{arm}.json"
        generation_path.write_text(json.dumps(generation), encoding="utf-8")
        return module.FactorArm(
            key=arm,
            generation_path=generation_path,
            prediction_path=prediction,
            generation=generation,
            configuration=dict(generation["configuration"]),
            descriptors={},
        )

    def _receipt(self, root: Path) -> tuple[Path, dict[str, Path]]:
        files: dict[str, Path] = {}
        primary: dict[str, dict[str, object]] = {}
        primary["builder_script"] = module.describe_file(Path(module.__file__))
        state_source_dir = root / "state_source"
        (state_source_dir / "src").mkdir(parents=True)
        (state_source_dir / "src" / "runtime.py").write_text(
            "# STATE runtime\n", encoding="utf-8"
        )
        for command in (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "test@example.invalid"),
            ("git", "config", "user.name", "Contract Test"),
            ("git", "add", "src/runtime.py"),
            ("git", "commit", "-q", "-m", "runtime"),
        ):
            subprocess.run(command, cwd=state_source_dir, check=True)
        state_source = module.authenticate_state_source(state_source_dir)
        generations: dict[str, dict[str, object]] = {}
        for arm in module.ARM_KEYS:
            prediction = root / f"{arm}_prediction.h5ad"
            prediction.write_bytes(f"prediction {arm}".encode("utf-8"))
            prediction_descriptor = module.describe_file(prediction)
            generation = self._generation(arm, prediction_descriptor)
            generation["provenance"]["state_source_commit"] = state_source[
                "repository_commit"
            ]
            generation_path = root / f"{arm}_generation.json"
            generation_path.write_text(json.dumps(generation), encoding="utf-8")
            files[f"{arm}_prediction_h5ad"] = prediction
            files[f"{arm}_generation_json"] = generation_path
            primary[f"{arm}_prediction_h5ad"] = prediction_descriptor
            primary[f"{arm}_generation_json"] = module.describe_file(generation_path)
            generations[arm] = generation

        manifest = root / "manifest.csv"
        manifest.write_text(
            "target_gene,validation_route\nD0,direct\nD1,direct\nH0,held_target\nH1,held_target\n",
            encoding="utf-8",
        )
        files["manifest"] = manifest
        primary["manifest"] = module.describe_file(manifest)

        v51 = root / "v51_scoring_receipt.json"
        v51.write_text(
            json.dumps(
                {
                    "schema": "vcc-public-validation-scoring-receipt-v1",
                    "status": "scoring_complete",
                    "context": "HepG2",
                    "predictions": {
                        "anchor": {
                            "path": primary["A_prediction_h5ad"]["path"],
                            "sha256": primary["A_prediction_h5ad"]["sha256"],
                        },
                        "v51_alpha010": {
                            "path": primary["B_prediction_h5ad"]["path"],
                            "sha256": primary["B_prediction_h5ad"]["sha256"],
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        files["v51_scoring_receipt"] = v51
        primary["v51_scoring_receipt"] = module.describe_file(v51)

        for arm in ("C", "D"):
            planned = {
                key: generations[arm]["configuration"][key]
                for key in (
                    "context",
                    "cells_per_group",
                    "seed",
                    "target_policy",
                    "target_remaining_fraction",
                    "residual_alpha",
                )
            }
            spec = root / f"{arm}_spec.json"
            spec.write_text(
                json.dumps(
                    {
                        "schema": module.SPEC_SCHEMA,
                        "output_tag": module.ARM_IDENTITIES[arm],
                        "configuration": planned,
                    }
                ),
                encoding="utf-8",
            )
            verification = root / f"{arm}_verification.json"
            verification_payload = {
                "schema": module.VERIFICATION_SCHEMA,
                "status": "passed",
                "validation_mode": "strict-v2",
                "output_tag": module.ARM_IDENTITIES[arm],
                "configuration": planned,
                "checks": {
                    "generator_inputs_authenticated": True,
                    "strict_generator_provenance_authenticated": True,
                },
                "provenance": {
                    "spec": module.describe_file(spec),
                    "generation_json": primary[f"{arm}_generation_json"],
                    "prediction_h5ad": primary[f"{arm}_prediction_h5ad"],
                },
            }
            verification.write_text(json.dumps(verification_payload), encoding="utf-8")
            files[f"{arm}_spec"] = spec
            files[f"{arm}_generation_verification"] = verification
            primary[f"{arm}_spec"] = module.describe_file(spec)
            primary[f"{arm}_generation_verification"] = module.describe_file(verification)

        shared: dict[str, dict[str, object]] = {}
        for name in module.V51_SHARED_INPUT_MAP:
            path = root / f"shared_{name}.bin"
            path.write_bytes(name.encode("utf-8"))
            shared[name] = module.describe_file(path)
        residual = root / "residual.npz"
        residual.write_bytes(b"residual")
        residual_json = root / "residual.json"
        residual_json.write_text("{}", encoding="utf-8")
        residual_descriptor = module.describe_file(residual)
        residual_json_descriptor = module.describe_file(residual_json)
        direct_helper = root / "generate_state_direct_counts.py"
        direct_helper.write_text("# direct-count helper\n", encoding="utf-8")
        inference_helper = root / "infer_state_effect_prior.py"
        inference_helper.write_text("# inference helper\n", encoding="utf-8")
        direct_helper_descriptor = module.describe_file(direct_helper)
        inference_helper_descriptor = module.describe_file(inference_helper)
        files["generate_state_direct_counts_helper"] = direct_helper
        files["infer_state_effect_prior_helper"] = inference_helper

        # Rebuild the synthetic small artifacts with the same cross-schema
        # descriptor mappings enforced by a real factor receipt.
        for arm in module.ARM_KEYS:
            generation = generations[arm]
            for normalized_name, report_name in module.V51_SHARED_INPUT_MAP.items():
                generation["provenance"][report_name] = shared[normalized_name]
            generation["provenance"]["residual_artifact"] = (
                residual_descriptor if arm in ("B", "D") else None
            )
            path = files[f"{arm}_generation_json"]
            path.write_text(json.dumps(generation), encoding="utf-8")
            primary[f"{arm}_generation_json"] = module.describe_file(path)
        for arm in ("C", "D"):
            spec_path = files[f"{arm}_spec"]
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            spec["inputs"] = {
                **shared,
                "residual_npz": residual_descriptor,
                "residual_json": residual_json_descriptor,
                "generate_state_direct_counts_helper": direct_helper_descriptor,
                "infer_state_effect_prior_helper": inference_helper_descriptor,
            }
            spec["state_source"] = state_source
            spec["provenance_contract"] = {
                "schema": module.STRICT_PROVENANCE_CONTRACT,
                "mode": "strict",
                "generator_report_provenance_authenticated": True,
                "optional_state_config_bound": True,
            }
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            primary[f"{arm}_spec"] = module.describe_file(spec_path)
            verification_path = files[f"{arm}_generation_verification"]
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["provenance"]["spec"] = primary[f"{arm}_spec"]
            verification["provenance"]["generation_json"] = primary[
                f"{arm}_generation_json"
            ]
            verification["provenance"]["authenticated_strict_inputs"] = {
                name: spec["inputs"][name]
                for name in (
                    "generator_script",
                    "generate_state_direct_counts_helper",
                    "infer_state_effect_prior_helper",
                    "perturbation_map",
                    "var_dims",
                    "state_config",
                )
            }
            verification["provenance"]["authenticated_state_source"] = state_source
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            primary[f"{arm}_generation_verification"] = module.describe_file(
                verification_path
            )
        held_hash = "a" * 64
        payload = {
            "schema": module.SCHEMA,
            "status": "passed",
            "context": "HepG2",
            "factor_grid": {
                "levels": module.FACTOR_LEVELS,
                "arm_identities": module.ARM_IDENTITIES,
                "complete_two_by_two_grid": True,
            },
            "contract": {
                "strict_v2_required_for_c_and_d": True,
                "only_registered_factor_differences": True,
                "all_shared_generator_inputs_identical": True,
                "strict_cd_executable_helper_descriptors_identical": True,
                "all_axes_and_source_control_pairing_identical": True,
                "cd_held_target_counts_identical": True,
                "factor_builder_did_not_consume_treated_expression_values": True,
            },
            "cd_count_identity": {
                "held_target_hard_gate_passed": True,
                "comparison": {
                    "held_target": {
                        "exact_count_identity": True,
                        "differing_rows": 0,
                        "differing_entries": 0,
                        "left_logical_csr_sha256": held_hash,
                    }
                },
            },
            "provenance": {
                "primary_artifacts": primary,
                "shared_generator_inputs": shared,
                "residual_npz": residual_descriptor,
                "strict_only_inputs": {
                    "residual_json": residual_json_descriptor,
                    "generate_state_direct_counts_helper": direct_helper_descriptor,
                    "infer_state_effect_prior_helper": inference_helper_descriptor,
                },
                "strict_state_source": state_source,
                "noncausal_main_repository_commits_by_arm": {
                    arm: generations[arm]["provenance"].get("repository_commit")
                    for arm in module.ARM_KEYS
                },
                "legacy_ab_helper_provenance": {
                    "helper_descriptors_present_in_historical_reports": False,
                    "compatibility_mode": "explicit-read-only-pre-strict-v2",
                    "strict_cd_helpers_must_not_be_attributed_retroactively_to_ab": True,
                },
            },
        }
        receipt = root / "factor_receipt.json"
        receipt.write_text(json.dumps(payload), encoding="utf-8")
        files["receipt"] = receipt
        return receipt, files

    def test_registered_two_by_two_grid_passes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {arm: self._arm(root, arm) for arm in module.ARM_KEYS}
            result = module.validate_registered_factor_grid(arms)
        self.assertTrue(result["complete_two_by_two_grid"])
        self.assertEqual(result["arm_identities"], module.ARM_IDENTITIES)

    def test_unregistered_configuration_drift_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arms = {arm: self._arm(root, arm) for arm in module.ARM_KEYS}
            arms["D"] = self._arm(root, "D", seed=7)
            with self.assertRaisesRegex(RuntimeError, "Unregistered generation configuration drift"):
                module.validate_registered_factor_grid(arms)

    def test_receipt_validator_rehashes_and_rederives_small_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, files = self._receipt(Path(directory))
            result = module.authenticate_factor_contract_receipt(
                receipt,
                expected_manifest=files["manifest"],
                expected_v51_scoring_receipt=files["v51_scoring_receipt"],
                expected_c_spec=files["C_spec"],
                expected_d_spec=files["D_spec"],
            )
        self.assertEqual(result["status"], "authenticated")
        self.assertEqual(result["factor_levels"], module.FACTOR_LEVELS)

    def test_receipt_validator_rejects_post_receipt_artifact_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, files = self._receipt(Path(directory))
            with files["C_generation_json"].open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "size differs"):
                module.authenticate_factor_contract_receipt(receipt)

    def test_receipt_validator_rejects_either_swapped_executable_helper(self) -> None:
        for helper_name in (
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
        ):
            with self.subTest(helper_name=helper_name), tempfile.TemporaryDirectory() as directory:
                receipt, files = self._receipt(Path(directory))
                files[helper_name].write_text(
                    "# adversarial helper replacement\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "(size|SHA-256) differs"):
                    module.authenticate_factor_contract_receipt(receipt)

    def test_receipt_validator_rejects_wrong_expected_spec_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, _ = self._receipt(Path(directory))
            wrong = Path(directory) / "wrong.json"
            wrong.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "path mismatch: C_spec"):
                module.authenticate_factor_contract_receipt(
                    receipt, expected_c_spec=wrong
                )

    def test_generation_validation_rejects_truthy_non_boolean_checks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = self._arm(Path(directory), "A").generation
            report["validation"]["checks"] = {"shape": "passed"}
            with self.assertRaisesRegex(RuntimeError, "generation validation did not pass"):
                module._validate_generation_common(report, arm="A")

    def test_receipt_validator_rejects_truthy_non_boolean_verification_check(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, files = self._receipt(Path(directory))
            verification_path = files["C_generation_verification"]
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["checks"]["generator_inputs_authenticated"] = "passed"
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            payload["provenance"]["primary_artifacts"][
                "C_generation_verification"
            ] = module.describe_file(verification_path)
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "strict verification differs"):
                module.authenticate_factor_contract_receipt(receipt)

    def test_main_repository_commit_only_delta_is_noncausal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt, files = self._receipt(Path(directory))
            generation_path = files["C_generation_json"]
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            generation["provenance"]["repository_commit"] = "f" * 40
            generation_path.write_text(json.dumps(generation), encoding="utf-8")
            verification_path = files["C_generation_verification"]
            verification = json.loads(verification_path.read_text(encoding="utf-8"))
            verification["provenance"]["generation_json"] = module.describe_file(
                generation_path
            )
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            primary = payload["provenance"]["primary_artifacts"]
            primary["C_generation_json"] = module.describe_file(generation_path)
            primary["C_generation_verification"] = module.describe_file(verification_path)
            payload["provenance"]["noncausal_main_repository_commits_by_arm"]["C"] = "f" * 40
            receipt.write_text(json.dumps(payload), encoding="utf-8")
            result = module.authenticate_factor_contract_receipt(receipt)
            self.assertEqual(result["status"], "authenticated")


if __name__ == "__main__":
    unittest.main()
