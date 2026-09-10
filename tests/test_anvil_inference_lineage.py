"""Focused tests for task-correlated Anvil reconstruction lineage."""

from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import anvil_inference_lineage as lineage  # noqa: E402


def valid_runtime() -> dict[str, object]:
    return {
        "schema": lineage.TRAINING_RUNTIME_SCHEMA,
        "run_name": "state_sm_anvil_20k_seed42_v1",
        "registered_seed": 42,
        "training_devices": 1,
        "parallelism": lineage.TRAINING_PARALLELISM,
        "wandb_enabled": False,
        "slurm": {
            "job_id": "20380001",
            "array_job_id": "20380000",
            "array_task_id": "0",
            "array_task_count": "4",
        },
        "source": {
            "checkpoint_archive_shim": {
                "callback_state_schema": lineage.ARCHIVE_CALLBACK_SCHEMA,
            }
        },
    }


def valid_environment() -> dict[str, str]:
    return {
        "VCC_TRAINING_ARRAY_JOB_ID": "20380000",
        "SLURM_JOB_ID": "20390001",
        "SLURM_ARRAY_JOB_ID": "20390000",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_ARRAY_TASK_COUNT": "4",
        "SLURM_JOB_DEPENDENCY": "aftercorr:20380000",
    }


class CorrelatedLineageTests(unittest.TestCase):
    def validate(
        self,
        runtime: dict[str, object] | None = None,
        environment: dict[str, str] | None = None,
    ) -> dict[str, str]:
        return lineage.validate_correlated_training_runtime(
            runtime or valid_runtime(),
            environment or valid_environment(),
            expected_seed=42,
            expected_run_name="state_sm_anvil_20k_seed42_v1",
        )

    def test_accepts_exact_aftercorr_and_correlated_v2_runtime(self) -> None:
        result = self.validate()
        self.assertEqual(result["training_array_job_id"], "20380000")
        self.assertEqual(result["training_array_task_id"], "0")
        self.assertEqual(result["inference_array_task_id"], "0")
        self.assertEqual(result["slurm_job_dependency"], "aftercorr:20380000")

    def test_rejects_wrong_or_weakened_dependency(self) -> None:
        for dependency in (
            None,
            "afterok:20380000",
            "aftercorr:20380002",
            "aftercorr:20380000?afterany:9",
            "aftercorr:20380000,afterany:9",
            "aftercorr:20380000:20380002",
        ):
            with self.subTest(dependency=dependency):
                environment = valid_environment()
                if dependency is None:
                    environment.pop("SLURM_JOB_DEPENDENCY")
                else:
                    environment["SLURM_JOB_DEPENDENCY"] = dependency
                with self.assertRaisesRegex(
                    lineage.LineageError, "SLURM_JOB_DEPENDENCY must be exactly"
                ):
                    self.validate(environment=environment)

    def test_rejects_mismatched_declared_job_or_correlated_task(self) -> None:
        mutations = {
            "declared training array job ID": ("slurm", "array_job_id", "20380002"),
            "not correlated": ("slurm", "array_task_id", "1"),
            "upstream array task count": ("slurm", "array_task_count", "3"),
        }
        for expected_error, (section, key, value) in mutations.items():
            with self.subTest(key=key):
                runtime = copy.deepcopy(valid_runtime())
                runtime[section][key] = value  # type: ignore[index]
                with self.assertRaisesRegex(lineage.LineageError, expected_error):
                    self.validate(runtime=runtime)

    def test_rejects_v1_runtime_and_unregistered_archive_shim(self) -> None:
        runtime = valid_runtime()
        runtime["schema"] = "vcc-anvil-state-training-runtime-v1"
        with self.assertRaisesRegex(lineage.LineageError, "runtime schema"):
            self.validate(runtime=runtime)

        runtime = valid_runtime()
        runtime["source"]["checkpoint_archive_shim"][  # type: ignore[index]
            "callback_state_schema"
        ] = "wrong"
        with self.assertRaisesRegex(lineage.LineageError, "callback schema"):
            self.validate(runtime=runtime)

    def test_rejects_noncanonical_job_ids_and_wrong_array_size(self) -> None:
        for key, value in (
            ("VCC_TRAINING_ARRAY_JOB_ID", "020380000"),
            ("SLURM_ARRAY_JOB_ID", "2039_0000"),
            ("SLURM_ARRAY_TASK_COUNT", "3"),
        ):
            with self.subTest(key=key):
                environment = valid_environment()
                environment[key] = value
                with self.assertRaises(lineage.LineageError):
                    self.validate(environment=environment)


class CheckpointArgminTests(unittest.TestCase):
    def test_requires_earliest_finite_argmin_and_matching_archive(self) -> None:
        validations = [
            {"global_step": 20, "val_loss": 0.2},
            {"global_step": 10, "val_loss": 0.2},
            {"global_step": 30, "val_loss": 0.4},
        ]
        archives = [
            {
                "global_step": row["global_step"],
                "val_loss": row["val_loss"],
                "checkpoint": f"/run/checkpoints/step{row['global_step']:05d}.ckpt",
            }
            for row in validations
        ]
        selected = {
            "global_step": 10,
            "val_loss": 0.2,
            "checkpoint": "/run/checkpoints/step00010.ckpt",
        }
        result = lineage.validate_selected_earliest_argmin(
            validations,
            archives,
            selected,
            expected_steps=(10, 20, 30),
        )
        self.assertEqual(result, selected)

    def test_rejects_valid_archive_that_is_not_argmin(self) -> None:
        validations = [
            {"global_step": 10, "val_loss": 0.4},
            {"global_step": 20, "val_loss": 0.1},
        ]
        archives = [
            {"global_step": 10, "val_loss": 0.4, "checkpoint": "/run/step00010.ckpt"},
            {"global_step": 20, "val_loss": 0.1, "checkpoint": "/run/step00020.ckpt"},
        ]
        with self.assertRaisesRegex(lineage.LineageError, "not the earliest finite"):
            lineage.validate_selected_earliest_argmin(
                validations,
                archives,
                {
                    "global_step": 10,
                    "val_loss": 0.4,
                    "checkpoint": "/run/step00010.ckpt",
                },
                expected_steps=(10, 20),
            )

    def test_rejects_archive_row_loss_or_path_mismatch(self) -> None:
        validations = [{"global_step": 10, "val_loss": 0.1}]
        selected = {
            "global_step": 10,
            "val_loss": 0.1,
            "checkpoint": "/run/step00010.ckpt",
        }
        with self.assertRaisesRegex(lineage.LineageError, "archive loss differs"):
            lineage.validate_selected_earliest_argmin(
                validations,
                [{"global_step": 10, "val_loss": 0.2, "checkpoint": "/run/step00010.ckpt"}],
                selected,
                expected_steps=(10,),
            )
        with self.assertRaisesRegex(lineage.LineageError, "checkpoint differs"):
            lineage.validate_selected_earliest_argmin(
                validations,
                [{"global_step": 10, "val_loss": 0.1, "checkpoint": "/run/other.ckpt"}],
                selected,
                expected_steps=(10,),
            )


class LauncherStaticContractTests(unittest.TestCase):
    def test_v1_launcher_is_truth_blind_and_requests_one_h100_per_task(self) -> None:
        launcher = (
            REPOSITORY
            / "slurm"
            / "anvil_h100_generate_state_hepg2_reconstruction_array_v1.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-3%4", launcher)
        self.assertIn("#SBATCH --gpus-per-node=1", launcher)
        self.assertIn('readonly GENERATOR_SEED=20260901', launcher)
        self.assertIn('readonly PUBLIC_ROOT="$PROJECT_DIR/artifacts/anvil_state_reconstruction"', launcher)
        self.assertIn('readonly OUTPUT_ROOT="$PUBLIC_ROOT/candidates/hepg2"', launcher)
        self.assertIn('state_sm_anvil_20k_seed${TRAIN_SEED}_v1', launcher)
        self.assertIn('state_sm_anvil_reconstructed_hepg2_seed${TRAIN_SEED}_v1', launcher)
        self.assertIn("vcc-anvil-state-reconstructed-inference-runtime-v2", launcher)
        self.assertIn("anvil_candidate_completion.json", launcher)
        self.assertIn("write_anvil_candidate_completion.py", launcher)
        self.assertIn("#SBATCH --no-requeue", launcher)
        runtime_args = launcher.split("readonly -a RUNTIME_ARGS=(", 1)[1].split(
            "\n)", 1
        )[0]
        self.assertEqual(runtime_args.count('\"$GENERATOR_SEED\"'), 1)
        self.assertNotIn("treated_truth", launcher.lower())
        self.assertNotIn("submission.zip", launcher)


if __name__ == "__main__":
    unittest.main()
