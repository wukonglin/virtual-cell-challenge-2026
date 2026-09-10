"""Tests for the final Anvil candidate-completion trust anchor."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import write_anvil_candidate_completion as completion  # noqa: E402


OUTPUT_TAG = "state_sm_anvil_reconstructed_hepg2_seed42_v1"


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def descriptor(path: Path) -> dict[str, object]:
    return completion._regular_file_descriptor(path, path.name)


class CompletionFixture:
    def __init__(self, root: Path) -> None:
        self.candidate_dir = root / "candidates" / "hepg2" / OUTPUT_TAG
        self.candidate_dir.mkdir(parents=True)
        self.selection_path = root / "state_run" / "checkpoints" / "selected_checkpoint.json"
        self.runtime_path = self.candidate_dir / "anvil_inference_runtime.json"
        self.spec_path = self.candidate_dir / "spec.json"
        self.generation_path = self.candidate_dir / "generation.json"
        self.prediction_path = self.candidate_dir / "prediction.h5ad"
        self.verification_path = self.candidate_dir / "generation_verified.json"
        self.output_path = self.candidate_dir / "anvil_candidate_completion.json"
        self.environment = {
            "SLURM_JOB_ID": "20390001",
            "SLURM_ARRAY_JOB_ID": "20390000",
            "SLURM_ARRAY_TASK_ID": "0",
            "SLURM_JOB_DEPENDENCY": "aftercorr:20380000",
            "VCC_TRAINING_ARRAY_JOB_ID": "20380000",
        }

        validation = [
            {
                "global_step": step,
                "val_loss": 0.1 if step == 4_000 else 1.0 + step / 100_000,
            }
            for step in completion.EXPECTED_VALIDATION_STEPS
        ]
        archives = [
            {
                **row,
                "checkpoint": str(
                    (root / "state_run" / "checkpoints" / f"step{row['global_step']:05d}.ckpt").resolve()
                ),
            }
            for row in validation
        ]
        selected = {**archives[1], "checkpoint_size_bytes": 123, "checkpoint_sha256": "a" * 64}
        write_json(
            self.selection_path,
            {
                "schema": completion.SELECTION_SCHEMA,
                "checkpoint_layout": "validation_step_archives",
                "unarchived_validation_steps": [],
                "validation_results": validation,
                "archived_candidates": archives,
                "selected": selected,
            },
        )
        selection_descriptor = descriptor(self.selection_path)

        self.prediction_path.write_bytes(b"synthetic-h5ad-content")
        prediction_descriptor = descriptor(self.prediction_path)

        write_json(
            self.spec_path,
            {
                "schema": completion.SPEC_SCHEMA,
                "output_tag": OUTPUT_TAG,
                "context": completion.EXPECTED_CONTEXT,
                "firewall": {
                    "generator_inputs_include_sealed_truth": False,
                    "treated_profiles_read_while_planning": False,
                },
                "checkpoint_selection": {"global_step": 4_000, "val_loss": 0.1},
                "inputs": {"selection_json": selection_descriptor},
            },
        )
        spec_descriptor = descriptor(self.spec_path)

        write_json(
            self.generation_path,
            {
                "schema": completion.GENERATION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "configuration": {"context": completion.EXPECTED_CONTEXT},
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "checkpoint_selection": {"global_step": 4_000, "val_loss": 0.1},
                "provenance": {
                    "output_h5ad": prediction_descriptor,
                    "checkpoint_selection": selection_descriptor,
                },
            },
        )
        generation_descriptor = descriptor(self.generation_path)

        write_json(
            self.verification_path,
            {
                "schema": completion.VERIFICATION_SCHEMA,
                "status": "passed",
                "output_tag": OUTPUT_TAG,
                "context": completion.EXPECTED_CONTEXT,
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
                    "spec": spec_descriptor,
                    "generation_json": generation_descriptor,
                    "prediction_h5ad": prediction_descriptor,
                },
            },
        )

        write_json(
            self.runtime_path,
            {
                "schema": completion.INFERENCE_RUNTIME_SCHEMA,
                "output_tag": OUTPUT_TAG,
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
                    "selection_json": selection_descriptor,
                },
            },
        )

    def build(self) -> dict[str, object]:
        return completion.build_completion_payload(
            runtime_path=self.runtime_path,
            selection_path=self.selection_path,
            spec_path=self.spec_path,
            generation_path=self.generation_path,
            prediction_path=self.prediction_path,
            verification_path=self.verification_path,
            output_tag=OUTPUT_TAG,
            context=completion.EXPECTED_CONTEXT,
            environment=self.environment,
            producer_path=Path(completion.__file__).resolve(),
        )


class CompletionReceiptTests(unittest.TestCase):
    def test_binds_six_artifacts_lineage_and_truth_firewall(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            payload = fixture.build()
            self.assertEqual(payload["schema"], completion.SCHEMA)
            self.assertEqual(payload["artifact_count"], 6)
            self.assertEqual(set(payload["artifacts"]), {
                "inference_runtime",
                "checkpoint_selection",
                "candidate_spec",
                "generation_report",
                "prediction_h5ad",
                "generation_verification",
            })
            self.assertTrue(
                payload["correlated_slurm_lineage"]["corresponding_array_task_authenticated"]
            )
            self.assertFalse(payload["truth_firewall"]["sealed_treated_profiles_read"])
            binding = {key: value for key, value in payload.items() if key != "binding"}
            self.assertEqual(
                payload["binding"]["coverage"],
                "every_top_level_field_except_binding",
            )
            self.assertEqual(
                payload["binding"]["canonical_json_sha256"],
                completion.canonical_binding_sha256(binding),
            )

    def test_atomic_completion_never_overwrites(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            payload = fixture.build()
            completion.atomic_write_json_no_overwrite(fixture.output_path, payload)
            original = fixture.output_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                completion.atomic_write_json_no_overwrite(fixture.output_path, payload)
            self.assertEqual(fixture.output_path.read_bytes(), original)

    def test_rejects_prediction_changed_after_verification(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            fixture.prediction_path.write_bytes(b"changed-after-verification")
            with self.assertRaisesRegex(completion.CompletionError, "prediction.*mismatch"):
                fixture.build()

    def test_rejects_lineage_or_truth_firewall_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            fixture.environment["SLURM_JOB_DEPENDENCY"] = "afterok:20380000"
            with self.assertRaisesRegex(completion.CompletionError, "exact aftercorr"):
                fixture.build()

        with tempfile.TemporaryDirectory() as temporary:
            fixture = CompletionFixture(Path(temporary))
            generation = json.loads(fixture.generation_path.read_text(encoding="utf-8"))
            generation["data_firewall"]["truth_inputs_read"] = ["forbidden"]
            write_json(fixture.generation_path, generation)
            with self.assertRaises(completion.CompletionError):
                fixture.build()


if __name__ == "__main__":
    unittest.main()
