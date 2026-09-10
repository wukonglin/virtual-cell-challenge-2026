"""Tests for authenticated, truth-blind Anvil candidate relocation."""

from __future__ import annotations

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import materialize_public_candidate_transport as module  # noqa: E402


class CandidateTransportTests(unittest.TestCase):
    context = "HepG2"
    prefix = "hepg2"
    tag = "state_sm_anvil_reconstructed_hepg2_seed42_v1"

    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @staticmethod
    def _remote_descriptor(path: Path, remote_path: str) -> dict:
        descriptor = module.describe_file(path)
        descriptor["path"] = remote_path
        return descriptor

    @classmethod
    def _remote_completion_descriptor(cls, path: Path, remote_path: str) -> dict:
        descriptor = cls._remote_descriptor(path, remote_path)
        descriptor["mtime_ns"] = path.stat().st_mtime_ns
        return descriptor

    def _fixture(self, root: Path) -> dict[str, Path | str]:
        project = root / "project"
        controls = project / "dataset/public_v51/hepg2_controls_only.h5ad"
        panel_csv = project / "dataset/public_v51/split_manifest.csv"
        residual_npz = project / "artifacts/public_v51/hepg2_state_residual_top100_v1.npz"
        support_genes = project / "dataset/state_support/extracted/gene_names.csv"
        for path, content in (
            (controls, b"controls-only generator input"),
            (panel_csv, b"target_gene,validation_route\nT1,direct\n"),
            (residual_npz, b"truth-blind residual bytes"),
            (support_genes, b"T1\nG1\n"),
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)

        source_run = root / "remote_exports" / self.tag
        transport_dir = source_run / "transport"
        transport_dir.mkdir(parents=True)
        panel_sidecar = transport_dir / "split_manifest.json"
        panel_payload = {
            "schema": module.PANEL_SCHEMA,
            "selection_contract": {"effect_values_accessed": False},
            "provenance": {
                "output_csv": self._remote_descriptor(
                    panel_csv, "/anvil/projects/example/dataset/public_v51/split_manifest.csv"
                )
            },
            "transport_provenance": {
                "schema": module.V51_TRANSPORT_SCHEMA,
                "payload_bytes_preserved": True,
                "scientific_content_changed": False,
            },
        }
        self._write_json(panel_sidecar, panel_payload)
        residual_sidecar = transport_dir / "hepg2_state_residual_top100_v1.json"
        residual_payload = {
            "schema": module.RESIDUAL_SCHEMA,
            "context": self.context,
            "contract": {"recipient_treated_profiles_used": False},
            "provenance": {
                "output_npz": self._remote_descriptor(
                    residual_npz,
                    "/anvil/projects/example/artifacts/public_v51/"
                    "hepg2_state_residual_top100_v1.npz",
                )
            },
            "transport_provenance": {
                "schema": module.V51_TRANSPORT_SCHEMA,
                "payload_bytes_preserved": True,
                "scientific_content_changed": False,
            },
        }
        self._write_json(residual_sidecar, residual_payload)

        remote_root = f"/anvil/scratch/user/vcc/candidates/hepg2/{self.tag}"
        remote_model = "/anvil/scratch/user/vcc/state_sm_anvil_20k_seed42_v1"
        selection_path = transport_dir / "selected_checkpoint.json"
        validation_results = [
            {
                "global_step": step,
                "val_loss": 0.1 if step == 4_000 else 1.0 + step / 100_000,
            }
            for step in module.EXPECTED_VALIDATION_STEPS
        ]
        archived_candidates = [
            {
                **row,
                "checkpoint": f"{remote_model}/checkpoints/step{row['global_step']:05d}.ckpt",
            }
            for row in validation_results
        ]
        selected = {
            **archived_candidates[1],
            "checkpoint_size_bytes": 123,
            "checkpoint_sha256": "a" * 64,
        }
        self._write_json(
            selection_path,
            {
                "schema": module.SELECTION_SCHEMA,
                "checkpoint_layout": "validation_step_archives",
                "unarchived_validation_steps": [],
                "validation_results": validation_results,
                "archived_candidates": archived_candidates,
                "selected": selected,
            },
        )
        selection_descriptor = self._remote_descriptor(
            selection_path, f"{remote_model}/checkpoints/selected_checkpoint.json"
        )
        canonical_descriptors = {
            "controls_h5ad": self._remote_descriptor(
                controls, "/anvil/projects/example/dataset/public_v51/hepg2_controls_only.h5ad"
            ),
            "panel_manifest": self._remote_descriptor(
                panel_sidecar, f"{remote_root}/transport/split_manifest.json"
            ),
            "panel_csv": self._remote_descriptor(
                panel_csv, "/anvil/projects/example/dataset/public_v51/split_manifest.csv"
            ),
            "residual_npz": self._remote_descriptor(
                residual_npz,
                "/anvil/projects/example/artifacts/public_v51/"
                "hepg2_state_residual_top100_v1.npz",
            ),
            "residual_json": self._remote_descriptor(
                residual_sidecar,
                f"{remote_root}/transport/hepg2_state_residual_top100_v1.json",
            ),
            "support_genes": self._remote_descriptor(
                support_genes,
                "/anvil/projects/example/dataset/state_support/extracted/gene_names.csv",
            ),
        }
        opaque = {
            "path": "/anvil/projects/example/model/opaque.bin",
            "size_bytes": 1,
            "sha256": "a" * 64,
        }
        inputs = {
            **canonical_descriptors,
            "checkpoint": copy.deepcopy(opaque),
            "selection_json": copy.deepcopy(selection_descriptor),
            "generator_script": copy.deepcopy(opaque),
            "generate_state_direct_counts_helper": copy.deepcopy(opaque),
            "infer_state_effect_prior_helper": copy.deepcopy(opaque),
            "perturbation_map": copy.deepcopy(opaque),
            "var_dims": copy.deepcopy(opaque),
            "state_config": None,
        }
        configuration = {
            "context": self.context,
            "cells_per_group": 400,
            "residual_alpha": 0.10,
        }
        spec = {
            "schema": module.SPEC_SCHEMA,
            "output_tag": self.tag,
            "context": self.context,
            "context_prefix": self.prefix,
            "configuration": configuration,
            "axes": {"targets": 300, "native_genes": 18533},
            "inputs": inputs,
            "checkpoint_selection": {"global_step": 4_000, "val_loss": 0.1},
            "state_source": {
                "path": "/anvil/projects/example/state",
                "repository_commit": "1" * 40,
                "repository_tree": "2" * 40,
                "tracked_worktree_clean": True,
                "untracked_and_ignored_runtime_files_absent": True,
                "runtime_isolation": {
                    "mode": "authenticated-clean-worktree-no-bytecode-v1",
                    "pythonpath_first": "/anvil/projects/example/state/src",
                    "python_dont_write_bytecode": True,
                },
            },
            "firewall": {
                "generator_inputs_include_sealed_truth": False,
                "treated_profiles_read_while_planning": False,
            },
            "provenance_contract": {
                "schema": "vcc-public-generator-provenance-v2",
                "mode": "strict",
                "generator_report_provenance_authenticated": True,
                "optional_state_config_bound": True,
            },
        }
        spec_path = source_run / "spec.json"
        self._write_json(spec_path, spec)

        prediction = source_run / "prediction.h5ad"
        prediction.write_bytes(b"opaque prediction payload; never decoded by transport")
        generation_provenance = {
            "controls_h5ad": copy.deepcopy(inputs["controls_h5ad"]),
            "panel_manifest": copy.deepcopy(inputs["panel_manifest"]),
            "panel_csv": copy.deepcopy(inputs["panel_csv"]),
            "support_gene_axis": copy.deepcopy(inputs["support_genes"]),
            "checkpoint": copy.deepcopy(inputs["checkpoint"]),
            "checkpoint_selection": copy.deepcopy(inputs["selection_json"]),
            "script": copy.deepcopy(inputs["generator_script"]),
            "perturbation_map": copy.deepcopy(inputs["perturbation_map"]),
            "var_dims": copy.deepcopy(inputs["var_dims"]),
            "state_config": None,
            "residual_artifact": copy.deepcopy(inputs["residual_npz"]),
            "state_source_commit": "1" * 40,
            "output_h5ad": self._remote_descriptor(
                prediction, f"{remote_root}/prediction.h5ad"
            ),
        }
        generation = {
            "schema": module.GENERATION_SCHEMA,
            "artifact_type": "sealed_public_validation_prediction",
            "full_frozen_panel_contract": True,
            "data_firewall": {
                "sealed_treated_profiles_read": False,
                "truth_inputs_read": [],
            },
            "configuration": configuration,
            "checkpoint_selection": {"global_step": 4_000, "val_loss": 0.1},
            "validation": {
                "shape": [120000, 18533],
                "groups": 300,
                "failed_checks": [],
            },
            "scientific_qc": {
                "all_libraries_exact": True,
                "all_native_non_support_counts_exact": True,
                "target_knockdown_failures": [],
            },
            "residual": {"enabled": True, "alpha": 0.10, "artifact_read": True},
            "provenance": generation_provenance,
        }
        generation_path = source_run / "generation.json"
        self._write_json(generation_path, generation)
        verification = {
            "schema": module.VERIFICATION_SCHEMA,
            "status": "passed",
            "output_tag": self.tag,
            "context": self.context,
            "configuration": configuration,
            "validation_mode": "strict-v2",
            "checks": {
                "generator_inputs_authenticated": True,
                "generator_did_not_read_treated_truth": True,
                "generation_matches_locked_configuration": True,
                "prediction_hash_matches_report": True,
                "prediction_shape_and_groups_match_spec": True,
                "exact_library_and_non_state_invariants_passed": True,
                "strict_generator_provenance_authenticated": True,
            },
            "provenance": {
                "spec": self._remote_descriptor(spec_path, f"{remote_root}/spec.json"),
                "generation_json": self._remote_descriptor(
                    generation_path, f"{remote_root}/generation.json"
                ),
                "prediction_h5ad": self._remote_descriptor(
                    prediction, f"{remote_root}/prediction.h5ad"
                ),
                "authenticated_strict_inputs": {},
                "authenticated_state_source": spec["state_source"],
            },
        }
        verification_path = source_run / "generation_verified.json"
        self._write_json(verification_path, verification)

        runtime_path = source_run / "anvil_inference_runtime.json"
        self._write_json(
            runtime_path,
            {
                "schema": module.INFERENCE_RUNTIME_SCHEMA,
                "output_tag": self.tag,
                "leaderboard_submission_authorized": False,
                "slurm": {
                    "job_id": "20390001",
                    "array_job_id": "20390000",
                    "array_task_id": "0",
                    "job_dependency": "aftercorr:20380000",
                    "declared_training_array_job_id": "20380000",
                    "correlated_upstream": {
                        "array_job_id": "20380000",
                        "array_task_id": "0",
                        "task_job_id": "20380001",
                        "dependency_type": "aftercorr",
                    },
                },
                "checkpoint_selection": {
                    "selected_step": 4_000,
                    "selected_val_loss": 0.1,
                    "checkpoint": copy.deepcopy(inputs["checkpoint"]),
                    "expected_checkpoint_sha256": inputs["checkpoint"]["sha256"],
                    "selection_json": selection_descriptor,
                },
                "transported_inputs": {
                    "panel_json": copy.deepcopy(inputs["panel_manifest"]),
                    "residual_json": copy.deepcopy(inputs["residual_json"]),
                },
            },
        )
        completion_artifacts = {
            "inference_runtime": self._remote_completion_descriptor(
                runtime_path, f"{remote_root}/anvil_inference_runtime.json"
            ),
            "checkpoint_selection": self._remote_completion_descriptor(
                selection_path, f"{remote_model}/checkpoints/selected_checkpoint.json"
            ),
            "candidate_spec": self._remote_completion_descriptor(
                spec_path, f"{remote_root}/spec.json"
            ),
            "generation_report": self._remote_completion_descriptor(
                generation_path, f"{remote_root}/generation.json"
            ),
            "prediction_h5ad": self._remote_completion_descriptor(
                prediction, f"{remote_root}/prediction.h5ad"
            ),
            "generation_verification": self._remote_completion_descriptor(
                verification_path, f"{remote_root}/generation_verified.json"
            ),
        }
        correlated_lineage = {
            "dependency": "aftercorr:20380000",
            "dependency_type": "aftercorr",
            "training_array_job_id": "20380000",
            "training_array_task_id": "0",
            "training_task_job_id": "20380001",
            "inference_array_job_id": "20390000",
            "inference_array_task_id": "0",
            "inference_task_job_id": "20390001",
            "corresponding_array_task_authenticated": True,
        }
        completion_document = {
            "schema": module.ANVIL_COMPLETION_SCHEMA,
            "output_tag": self.tag,
            "context": self.context,
            "candidate_directory": remote_root,
            "correlated_slurm_lineage": correlated_lineage,
            "truth_firewall": {
                "schema": module.ANVIL_FIREWALL_SCHEMA,
                "candidate_generation_truth_blind": True,
                "sealed_treated_profiles_read": False,
                "truth_inputs_read": [],
                "scoring_invoked_by_launcher": False,
                "leaderboard_submission_authorized": False,
            },
            "artifacts": completion_artifacts,
            "completed_utc": "2026-09-03T12:00:00+00:00",
            "completion_state": "complete_after_strict_generation_verification",
            "artifact_count": 6,
            "producer": {
                "path": "/anvil/projects/example/scripts/write_anvil_candidate_completion.py",
                "size_bytes": 10,
                "mtime_ns": 123456789,
                "sha256": "b" * 64,
            },
        }
        completion_payload = {
            **completion_document,
            "binding": {
                "algorithm": "sha256",
                "coverage": "every_top_level_field_except_binding",
                "canonical_json_sha256": module.canonical_binding_sha256(
                    completion_document
                ),
            },
        }
        completion_path = source_run / "anvil_candidate_completion.json"
        self._write_json(completion_path, completion_payload)
        return {
            "project": project,
            "source_run": source_run,
            "destination_root": project / "artifacts/anvil_state_reconstruction",
            "completion_sha": module.sha256_file(completion_path),
            "completion": completion_path,
            "runtime": runtime_path,
            "selection": selection_path,
            "prediction": prediction,
            "panel_sidecar": panel_sidecar,
            "residual_sidecar": residual_sidecar,
            "controls": controls,
            "panel_csv": panel_csv,
            "residual_npz": residual_npz,
            "support_genes": support_genes,
        }

    def _materialize(self, fixture: dict[str, Path | str]) -> Path:
        receipt = module.materialize(
            project_dir=Path(fixture["project"]),
            source_run=Path(fixture["source_run"]),
            source_completion_expected_sha256=str(fixture["completion_sha"]),
            destination_public_root=Path(fixture["destination_root"]),
            context=self.context,
            output_tag=self.tag,
        )
        return receipt.parent

    def test_materializes_path_only_candidate_and_validates_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            source_bytes = {
                name: Path(fixture[name]).read_bytes()
                for name in (
                    "completion",
                    "runtime",
                    "selection",
                    "prediction",
                    "panel_sidecar",
                    "residual_sidecar",
                )
            }
            destination = self._materialize(fixture)

            carried_destinations = {
                "completion": destination / "anvil_candidate_completion.json",
                "runtime": destination / "anvil_inference_runtime.json",
                "selection": destination / "transport/selected_checkpoint.json",
                "prediction": destination / "prediction.h5ad",
            }
            for name, carried in carried_destinations.items():
                self.assertTrue(
                    os.path.samestat(Path(fixture[name]).stat(), carried.stat())
                )
            self.assertEqual(
                (destination / "prediction.h5ad").read_bytes(),
                source_bytes["prediction"],
            )
            for name in source_bytes:
                self.assertEqual(Path(fixture[name]).read_bytes(), source_bytes[name])

            spec = json.loads((destination / "spec.json").read_text(encoding="utf-8"))
            generation = json.loads(
                (destination / "generation.json").read_text(encoding="utf-8")
            )
            verification = json.loads(
                (destination / "generation_verified.json").read_text(encoding="utf-8")
            )
            receipt = module.validate_transport_receipt(
                run_dir=destination,
                project_dir=Path(fixture["project"]),
                expected_context=self.context,
                expected_output_tag=self.tag,
            )
            self.assertEqual(receipt["schema"], module.TRANSPORT_RECEIPT_SCHEMA)
            self.assertTrue(receipt["contract"]["prediction_bytes_preserved"])
            self.assertTrue(
                receipt["contract"]["source_completion_canonical_binding_authenticated"]
            )
            self.assertTrue(
                receipt["contract"]["completion_bound_six_artifacts_authenticated"]
            )
            self.assertEqual(
                receipt["source"]["independently_supplied_completion_sha256"],
                fixture["completion_sha"],
            )
            self.assertEqual(
                set(receipt["destination"]["artifacts"]),
                {
                    "candidate_completion",
                    "inference_runtime",
                    "checkpoint_selection",
                    "spec",
                    "generation_json",
                    "prediction_h5ad",
                    "generation_verification",
                    "panel_manifest",
                    "residual_json",
                },
            )
            self.assertEqual(
                set(receipt["completion_bound_artifacts"]),
                set(module.COMPLETION_ARTIFACT_MAP),
            )
            self.assertFalse(receipt["contract"]["scientific_content_changed"])
            self.assertFalse(receipt["contract"]["treated_truth_read"])
            self.assertEqual(
                set(receipt["json_change_allowlist"]),
                {"spec.json", "generation.json", "generation_verified.json"},
            )
            self.assertEqual(
                receipt["regenerated_authentication_fields"],
                [
                    "/generation_verified.json/provenance/spec",
                    "/generation_verified.json/provenance/generation_json",
                ],
            )
            self.assertEqual(
                Path(spec["inputs"]["controls_h5ad"]["path"]),
                Path(fixture["controls"]).resolve(),
            )
            self.assertEqual(
                Path(spec["inputs"]["panel_manifest"]["path"]),
                destination / "transport/split_manifest.json",
            )
            self.assertEqual(
                Path(spec["inputs"]["selection_json"]["path"]),
                destination / "transport/selected_checkpoint.json",
            )
            self.assertEqual(
                generation["provenance"]["output_h5ad"]["sha256"],
                module.sha256_file(Path(fixture["prediction"])),
            )
            self.assertTrue(verification["checks"]["authenticated_candidate_transport"])

    def test_bad_external_trust_anchor_fails_without_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(RuntimeError, "independent trust anchor"):
                module.materialize(
                    project_dir=Path(fixture["project"]),
                    source_run=Path(fixture["source_run"]),
                    source_completion_expected_sha256="0" * 64,
                    destination_public_root=Path(fixture["destination_root"]),
                    context=self.context,
                    output_tag=self.tag,
                )
            destination = (
                Path(fixture["destination_root"])
                / "candidates"
                / self.prefix
                / self.tag
            )
            self.assertFalse(destination.exists())

    def test_completion_canonical_binding_is_recomputed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            completion_path = Path(fixture["completion"])
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
            completion["completed_utc"] = "2026-09-03T12:00:01+00:00"
            self._write_json(completion_path.with_suffix(".changed.json"), completion)
            completion_path.unlink()
            completion_path.with_suffix(".changed.json").rename(completion_path)
            fixture["completion_sha"] = module.sha256_file(completion_path)
            with self.assertRaisesRegex(RuntimeError, "canonical binding SHA-256"):
                self._materialize(fixture)

    def test_completion_bound_mtime_requires_archive_preserving_pull(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            selection = Path(fixture["selection"])
            current = selection.stat().st_mtime_ns
            os.utime(selection, ns=(current + 1_000_000_000, current + 1_000_000_000))
            with self.assertRaisesRegex(RuntimeError, "pull with rsync -a"):
                self._materialize(fixture)

    def test_canonical_input_drift_fails_before_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            Path(fixture["controls"]).write_bytes(b"swapped local content")
            with self.assertRaisesRegex(RuntimeError, "canonical local controls_h5ad"):
                self._materialize(fixture)
            destination = (
                Path(fixture["destination_root"])
                / "candidates"
                / self.prefix
                / self.tag
            )
            self.assertFalse(destination.exists())

    def test_source_bundle_tampering_is_rejected_by_completion_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            generation = Path(fixture["source_run"]) / "generation.json"
            generation.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                RuntimeError, "completion artifact generation_report"
            ):
                self._materialize(fixture)

    def test_destination_json_tampering_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            destination = self._materialize(fixture)
            generation_path = destination / "generation.json"
            generation = json.loads(generation_path.read_text(encoding="utf-8"))
            generation["scientific_qc"]["all_libraries_exact"] = False
            self._write_json(destination / "tampered.json", generation)
            generation_path.unlink()
            (destination / "tampered.json").rename(generation_path)
            with self.assertRaisesRegex(RuntimeError, "outside its allowlist"):
                module.validate_transport_receipt(
                    run_dir=destination,
                    project_dir=Path(fixture["project"]),
                    expected_context=self.context,
                    expected_output_tag=self.tag,
                )

    def test_missing_receipt_is_required_for_anvil_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            destination = self._materialize(fixture)
            (destination / module.TRANSPORT_RECEIPT_NAME).unlink()
            with self.assertRaisesRegex(RuntimeError, "Missing candidate transport receipt"):
                module.validate_transport_receipt(
                    run_dir=destination,
                    project_dir=Path(fixture["project"]),
                    expected_context=self.context,
                    expected_output_tag=self.tag,
                )

    def test_local_candidate_without_anvil_lineage_does_not_require_transport(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / "project"
            project.mkdir()
            run = root / "ordinary_local_candidate"
            run.mkdir()
            self._write_json(
                run / "spec.json",
                {"state_source": {"path": str(project / "external/state")}},
            )
            self.assertIsNone(
                module.validate_transport_receipt(
                    run_dir=run,
                    project_dir=project,
                    expected_context=self.context,
                    expected_output_tag="ordinary_local_candidate",
                )
            )

    def test_cli_has_no_truth_input_and_score_gate_precedes_truth_access(self) -> None:
        parser_text = Path(module.__file__).read_text(encoding="utf-8")
        self.assertNotIn('add_argument("--truth', parser_text)
        self.assertNotIn('add_argument("--sealed-truth', parser_text)
        launcher = (ROOT / "slurm/cpu_score_public_candidate_v6.sbatch").read_text(
            encoding="utf-8"
        )
        transport_gate = launcher.index('"$TRANSPORT_VERIFIER" verify')
        truth_access = launcher.index('if [[ ! -r "$SEALED_TRUTH" ]]')
        scorer = launcher.index('"$PYTHON_BIN" scripts/prepare_public_scoring_views.py')
        self.assertLess(transport_gate, truth_access)
        self.assertLess(truth_access, scorer)


if __name__ == "__main__":
    unittest.main()
