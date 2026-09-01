"""Tests for immutable V6 public-candidate planning and verification."""

from __future__ import annotations

import json
import os
import sys
import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import public_candidate_v6_contract as module  # noqa: E402


class PublicCandidateV6ContractTests(unittest.TestCase):
    def _inputs(self, root: Path) -> Namespace:
        targets = [f"T{index:03d}" for index in range(300)]
        genes = [*targets, "G000"]
        direct = np.asarray([True] * 267 + [False] * 33)

        panel_csv = root / "panel.csv"
        pd.DataFrame(
            {
                "target_gene": targets,
                "validation_route": np.where(direct, "direct", "held_target"),
            }
        ).to_csv(panel_csv, index=False)
        panel_json = root / "panel.json"
        panel_json.write_text(
            json.dumps(
                {
                    "schema": module.PANEL_SCHEMA,
                    "selection_contract": {"effect_values_accessed": False},
                    "provenance": {"output_csv": module.describe_file(panel_csv)},
                }
            ),
            encoding="utf-8",
        )

        controls = root / "controls.h5ad"
        obs = pd.DataFrame(
            {
                "context": pd.Categorical(["HepG2"] * 400),
                "target_gene": pd.Categorical([module.CONTROL_LABEL] * 400),
            },
            index=[f"control_{index}" for index in range(400)],
        )
        data = ad.AnnData(
            sp.csr_matrix((400, len(genes)), dtype=np.int32),
            obs=obs,
            var=pd.DataFrame(index=genes),
        )
        data.uns["public_validation"] = {
            "schema": module.DATA_SCHEMA,
            "role": "controls-only-generator-input",
            "sealed_treated_profiles_present": False,
        }
        data.write_h5ad(controls)

        residual_npz = root / "residual.npz"
        effects = np.zeros((1, 300, len(genes)), dtype=np.float32)
        effects[0, :267, -1] = np.linspace(0.1, 0.2, 267, dtype=np.float32)
        np.savez_compressed(
            residual_npz,
            effects=effects,
            target_names=np.asarray(targets),
            gene_names=np.asarray(genes),
            contexts=np.asarray(["HepG2"]),
            direct_mask=direct,
            fallback_mask=~direct,
        )
        residual_json = root / "residual.json"
        residual_json.write_text(
            json.dumps(
                {
                    "schema": module.RESIDUAL_SCHEMA,
                    "context": "HepG2",
                    "configuration": {"shared_response_weight": 0.0},
                    "contract": {
                        "held_target_rows_are_zero": True,
                        "recipient_treated_profiles_used": False,
                        "native_non_state_genes_are_immutable": True,
                    },
                    "provenance": {"output_npz": module.describe_file(residual_npz)},
                }
            ),
            encoding="utf-8",
        )

        support = root / "support.csv"
        support.write_text(
            "\n".join([*genes, *(f"S{index:05d}" for index in range(module.EXPECTED_SUPPORT_GENES - len(genes)))])
            + "\n",
            encoding="utf-8",
        )
        checkpoint = root / "selected.ckpt"
        checkpoint.write_bytes(b"test checkpoint")
        selection = root / "selection.json"
        selection.write_text(
            json.dumps(
                {
                    "schema": "vcc-state-checkpoint-selection-v1",
                    "selected": {
                        "checkpoint": str(checkpoint.resolve()),
                        "checkpoint_sha256": module.sha256_file(checkpoint),
                        "checkpoint_size_bytes": checkpoint.stat().st_size,
                        "global_step": 16000,
                        "val_loss": 1.5,
                    },
                    "selection_rule": "test",
                }
            ),
            encoding="utf-8",
        )
        generator_script = root / "generate.py"
        generator_script.write_text("# synthetic generator\n", encoding="utf-8")
        direct_counts_helper = root / "generate_state_direct_counts.py"
        direct_counts_helper.write_text("# synthetic direct-count helper\n", encoding="utf-8")
        inference_helper = root / "infer_state_effect_prior.py"
        inference_helper.write_text("# synthetic inference helper\n", encoding="utf-8")
        perturbation_map = root / "pert_onehot_map.pt"
        perturbation_map.write_bytes(b"synthetic perturbation map")
        var_dims = root / "var_dims.pkl"
        var_dims.write_bytes(b"synthetic dimensions")
        state_config = root / "config.yaml"
        state_config.write_text("model: synthetic\n", encoding="utf-8")
        state_source_dir = root / "state_source"
        (state_source_dir / "src").mkdir(parents=True)
        (state_source_dir / "src" / "runtime.py").write_text(
            "# synthetic STATE runtime\n", encoding="utf-8"
        )
        for command in (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "test@example.invalid"),
            ("git", "config", "user.name", "Contract Test"),
            ("git", "add", "src/runtime.py"),
            ("git", "commit", "-q", "-m", "synthetic runtime"),
        ):
            subprocess.run(command, cwd=state_source_dir, check=True)
        old_pythonpath = os.environ.get("PYTHONPATH")
        old_no_bytecode = os.environ.get("PYTHONDONTWRITEBYTECODE")
        os.environ["PYTHONPATH"] = str((state_source_dir / "src").resolve())
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        self.addCleanup(
            lambda: (
                os.environ.pop("PYTHONPATH", None)
                if old_pythonpath is None
                else os.environ.__setitem__("PYTHONPATH", old_pythonpath),
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
                if old_no_bytecode is None
                else os.environ.__setitem__("PYTHONDONTWRITEBYTECODE", old_no_bytecode),
            )
        )
        return Namespace(
            context="HepG2",
            output_tag="v51_p0_a0025",
            residual_alpha=0.025,
            target_remaining_fraction=0.20,
            controls_h5ad=controls,
            panel_manifest=panel_json,
            panel_csv=panel_csv,
            residual_npz=residual_npz,
            residual_json=residual_json,
            checkpoint=checkpoint,
            selection_json=selection,
            support_genes=support,
            generator_script=generator_script,
            generate_state_direct_counts_helper=direct_counts_helper,
            infer_state_effect_prior_helper=inference_helper,
            state_source_dir=state_source_dir,
            perturbation_map=perturbation_map,
            var_dims=var_dims,
            state_config=state_config,
        )

    @staticmethod
    def _generation_provenance(
        spec: dict[str, object], *, residual_enabled: bool
    ) -> dict[str, object]:
        inputs = spec["inputs"]
        assert isinstance(inputs, dict)
        return {
            "state_source_commit": spec["state_source"]["repository_commit"],
            "controls_h5ad": inputs["controls_h5ad"],
            "panel_manifest": inputs["panel_manifest"],
            "panel_csv": inputs["panel_csv"],
            "support_gene_axis": inputs["support_genes"],
            "checkpoint": inputs["checkpoint"],
            "checkpoint_selection": inputs["selection_json"],
            "script": inputs["generator_script"],
            "generate_state_direct_counts_helper": inputs[
                "generate_state_direct_counts_helper"
            ],
            "infer_state_effect_prior_helper": inputs[
                "infer_state_effect_prior_helper"
            ],
            "perturbation_map": inputs["perturbation_map"],
            "var_dims": inputs["var_dims"],
            "state_config": inputs["state_config"],
            "residual_artifact": inputs["residual_npz"] if residual_enabled else None,
        }

    def test_plan_authenticates_a_complete_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = module.build_spec(self._inputs(Path(directory)))
            self.assertEqual(spec["schema"], module.SPEC_SCHEMA)
            self.assertEqual(spec["context"], "HepG2")
            self.assertEqual(spec["configuration"]["residual_alpha"], 0.025)
            self.assertEqual(spec["configuration"]["target_remaining_fraction"], 0.20)
            self.assertEqual(spec["axes"]["direct_targets"], 267)
            self.assertFalse(spec["firewall"]["generator_inputs_include_sealed_truth"])
            self.assertEqual(spec["schema"], module.SPEC_SCHEMA)
            self.assertEqual(
                spec["provenance_contract"]["schema"],
                module.STRICT_PROVENANCE_CONTRACT,
            )
            self.assertEqual(
                spec["inputs"]["generator_script"]["sha256"],
                module.sha256_file(Path(directory) / "generate.py"),
            )

    def test_plan_rejects_helper_outside_generator_import_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self._inputs(root)
            decoy = root / "decoy" / "generate_state_direct_counts.py"
            decoy.parent.mkdir()
            decoy.write_bytes(args.generate_state_direct_counts_helper.read_bytes())
            args.generate_state_direct_counts_helper = decoy
            with self.assertRaisesRegex(RuntimeError, "generator import directory"):
                module.build_spec(args)

    def test_plan_rejects_dirty_tracked_state_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            (args.state_source_dir / "src" / "runtime.py").write_text(
                "# modified tracked runtime\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "tracked worktree changes"):
                module.build_spec(args)

    def test_plan_rejects_untracked_file_under_state_src(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            (args.state_source_dir / "src" / "injected.py").write_text(
                "# untracked importable runtime\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "import-affecting files"):
                module.build_spec(args)

    def test_plan_rejects_ignored_shared_object_or_pth_under_state_src(self) -> None:
        for suffix in (".so", ".pth"):
            with self.subTest(suffix=suffix), tempfile.TemporaryDirectory() as directory:
                args = self._inputs(Path(directory))
                (args.state_source_dir / ".gitignore").write_text(
                    f"*{suffix}\n", encoding="utf-8"
                )
                (args.state_source_dir / "src" / f"injected{suffix}").write_bytes(b"unsafe")
                with self.assertRaisesRegex(RuntimeError, "import-affecting files"):
                    module.build_spec(args)

    def test_plan_records_clean_source_and_no_bytecode_isolation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = module.build_spec(self._inputs(Path(directory)))
            isolation = spec["state_source"]["runtime_isolation"]
            self.assertEqual(
                isolation["mode"], "authenticated-clean-worktree-no-bytecode-v1"
            )
            self.assertTrue(isolation["python_dont_write_bytecode"])
            self.assertEqual(
                Path(isolation["pythonpath_first"]),
                Path(spec["state_source"]["path"]) / "src",
            )

    def test_plan_rejects_a_residual_changed_after_sidecar_creation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            with args.residual_npz.open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(RuntimeError, "residual NPZ size"):
                module.build_spec(args)

    def test_unsafe_tags_and_invalid_alpha_are_rejected(self) -> None:
        for value in ("../escape", "Uppercase", "contains space", ""):
            with self.subTest(value=value), self.assertRaisesRegex(RuntimeError, "Unsafe"):
                module.validate_output_tag(value)
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            for value in (-0.01, float("nan"), float("inf")):
                args.residual_alpha = value
                with self.subTest(value=value), self.assertRaisesRegex(
                    RuntimeError, "finite and nonnegative"
                ):
                    module.build_spec(args)

    def test_zero_alpha_state_only_plan_still_authenticates_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            args.residual_alpha = 0.0
            spec = module.build_spec(args)
            self.assertEqual(spec["configuration"]["residual_alpha"], 0.0)
            self.assertEqual(
                spec["inputs"]["residual_npz"]["sha256"],
                module.sha256_file(args.residual_npz),
            )

    def test_only_default_and_preregistered_p3_target_fractions_are_accepted(self) -> None:
        self.assertEqual(module.validate_target_remaining_fraction(0.20), 0.20)
        self.assertEqual(module.validate_target_remaining_fraction(0.40), 0.40)
        for value in (0.0, 0.30, 1.0, float("nan")):
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "Target remaining fraction"
            ):
                module.validate_target_remaining_fraction(value)

        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            args.target_remaining_fraction = 0.40
            spec = module.build_spec(args)
            self.assertEqual(spec["configuration"]["target_remaining_fraction"], 0.40)

    def test_generation_report_must_match_locked_alpha_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = module.build_spec(self._inputs(Path(directory)))
            provenance = self._generation_provenance(spec, residual_enabled=True)
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": spec["configuration"],
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": provenance,
            }
            module.validate_generation_report(report, spec)
            report["configuration"] = dict(report["configuration"])
            report["configuration"]["residual_alpha"] = 0.05
            with self.assertRaisesRegex(RuntimeError, "residual_alpha"):
                module.validate_generation_report(report, spec)

    def test_strict_generation_report_rejects_code_or_model_input_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = module.build_spec(self._inputs(Path(directory)))
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": spec["configuration"],
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": self._generation_provenance(
                    spec, residual_enabled=True
                ),
            }
            for key in (
                "script",
                "perturbation_map",
                "var_dims",
                "state_config",
            ):
                with self.subTest(key=key):
                    changed = json.loads(json.dumps(report))
                    changed["provenance"][key]["sha256"] = "0" * 64
                    with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                        module.validate_generation_report(changed, spec)

    def test_verification_rehashes_bound_generator_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan_args = self._inputs(root)
            spec = module.build_spec(plan_args)
            prediction = root / "prediction.h5ad"
            prediction.write_bytes(b"not reached because strict provenance fails first")
            provenance = self._generation_provenance(spec, residual_enabled=True)
            provenance["output_h5ad"] = module.describe_file(prediction)
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": spec["configuration"],
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": provenance,
            }
            spec_path = root / "spec.json"
            generation_path = root / "generation.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            generation_path.write_text(json.dumps(report), encoding="utf-8")

            original = plan_args.generator_script.read_text(encoding="utf-8")
            plan_args.generator_script.write_text(
                original.replace("synthetic", "mutatedxx"), encoding="utf-8"
            )
            verify_args = Namespace(
                spec=spec_path,
                prediction_h5ad=prediction,
                generation_json=generation_path,
                allow_legacy_v1_read_only=False,
            )
            with self.assertRaisesRegex(
                RuntimeError, "strict input generator_script SHA-256"
            ):
                module.verify_generation(verify_args)

    def test_verification_rejects_swapped_executable_helpers(self) -> None:
        for argument_name in (
            "generate_state_direct_counts_helper",
            "infer_state_effect_prior_helper",
        ):
            with self.subTest(argument_name=argument_name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                plan_args = self._inputs(root)
                spec = module.build_spec(plan_args)
                prediction = root / "prediction.h5ad"
                prediction.write_bytes(b"verification must fail before H5AD parsing")
                provenance = self._generation_provenance(spec, residual_enabled=True)
                provenance["output_h5ad"] = module.describe_file(prediction)
                report = {
                    "schema": module.PREDICTION_SCHEMA,
                    "artifact_type": "sealed_public_validation_prediction",
                    "full_frozen_panel_contract": True,
                    "data_firewall": {
                        "sealed_treated_profiles_read": False,
                        "truth_inputs_read": [],
                    },
                    "configuration": spec["configuration"],
                    "validation": {
                        "shape": [120000, spec["axes"]["native_genes"]],
                        "groups": 300,
                        "failed_checks": [],
                    },
                    "scientific_qc": {
                        "all_libraries_exact": True,
                        "all_native_non_support_counts_exact": True,
                        "target_knockdown_failures": [],
                    },
                    "residual": {
                        "enabled": True,
                        "alpha": 0.025,
                        "artifact_read": True,
                    },
                    "provenance": provenance,
                }
                spec_path = root / "spec.json"
                generation_path = root / "generation.json"
                spec_path.write_text(json.dumps(spec), encoding="utf-8")
                generation_path.write_text(json.dumps(report), encoding="utf-8")
                helper = Path(getattr(plan_args, argument_name))
                helper.write_text("# adversarial replacement helper\n", encoding="utf-8")
                with self.assertRaisesRegex(
                    RuntimeError, f"strict input {argument_name} (size|SHA-256)"
                ):
                    module.verify_generation(
                        Namespace(
                            spec=spec_path,
                            prediction_h5ad=prediction,
                            generation_json=generation_path,
                            allow_legacy_v1_read_only=False,
                        )
                    )

    def test_optional_state_config_presence_is_bound_both_ways(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            args.state_config = None
            spec = module.build_spec(args)
            self.assertIsNone(spec["inputs"]["state_config"])
            provenance = self._generation_provenance(spec, residual_enabled=True)
            self.assertIsNone(provenance["state_config"])
            provenance["state_config"] = module.describe_file(Path(directory) / "config.yaml")
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": spec["configuration"],
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": provenance,
            }
            with self.assertRaisesRegex(RuntimeError, "unbound STATE model config"):
                module.validate_generation_report(report, spec)

    def test_legacy_v1_requires_explicit_read_only_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = module.build_spec(self._inputs(Path(directory)))
            legacy = json.loads(json.dumps(spec))
            legacy["schema"] = module.LEGACY_SPEC_SCHEMA
            legacy.pop("provenance_contract")
            for key in (
                "generator_script",
                "generate_state_direct_counts_helper",
                "infer_state_effect_prior_helper",
                "perturbation_map",
                "var_dims",
                "state_config",
            ):
                legacy["inputs"].pop(key)
            provenance = self._generation_provenance(spec, residual_enabled=True)
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": legacy["configuration"],
                "validation": {
                    "shape": [120000, legacy["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": provenance,
            }
            with self.assertRaisesRegex(RuntimeError, "legacy v1 requires explicit"):
                module.validate_generation_report(report, legacy)
            module.validate_generation_report(
                report, legacy, allow_legacy_v1_read_only=True
            )

    def test_generation_report_must_match_locked_target_fraction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            args.target_remaining_fraction = 0.40
            spec = module.build_spec(args)
            provenance = self._generation_provenance(spec, residual_enabled=True)
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": {**spec["configuration"], "target_remaining_fraction": 0.20},
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": True,
                    "alpha": 0.025,
                    "artifact_read": True,
                },
                "provenance": provenance,
            }
            with self.assertRaisesRegex(RuntimeError, "target_remaining_fraction"):
                module.validate_generation_report(report, spec)

    def test_zero_alpha_generation_must_report_an_unread_bound_residual(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._inputs(Path(directory))
            args.residual_alpha = 0.0
            args.target_remaining_fraction = 0.40
            spec = module.build_spec(args)
            provenance = self._generation_provenance(spec, residual_enabled=False)
            report = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": spec["configuration"],
                "validation": {
                    "shape": [120000, spec["axes"]["native_genes"]],
                    "groups": 300,
                    "failed_checks": [],
                },
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": False,
                    "alpha": 0.0,
                    "artifact_read": False,
                    "declared_path": str(args.residual_npz.resolve()),
                },
                "provenance": provenance,
            }
            module.validate_generation_report(report, spec)

            report["residual"] = dict(report["residual"])
            report["residual"]["declared_path"] = str(Path(directory) / "other.npz")
            with self.assertRaisesRegex(RuntimeError, "declaration differs"):
                module.validate_generation_report(report, spec)


if __name__ == "__main__":
    unittest.main()
