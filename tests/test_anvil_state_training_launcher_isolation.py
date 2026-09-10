"""Static contracts for bytecode-isolated Anvil STATE training launchers."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
SMOKE = REPOSITORY / "slurm" / "anvil_h100_train_state_smoke_v1.sbatch"
PRODUCTION = REPOSITORY / "slurm" / "anvil_h100_train_state_sm_20k_array_v1.sbatch"
EXPECTED_STATE_COMMIT = "9bbfe78a434a55205e4de834e1ea99f85f7a3add"


def launcher_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    if not text.startswith("#!/usr/bin/env bash\n"):
        raise AssertionError(f"Launcher has no Bash shebang: {path}")
    return text


def directive(text: str, name: str) -> str:
    match = re.search(rf"^#SBATCH --{re.escape(name)}=(.+)$", text, re.MULTILINE)
    if match is None:
        raise AssertionError(f"Missing Slurm directive: {name}")
    return match.group(1)


def assignment(text: str, name: str) -> str:
    match = re.search(
        rf'^readonly {re.escape(name)}="([^"]+)"$', text, re.MULTILINE
    )
    if match is None:
        raise AssertionError(f"Missing readonly assignment: {name}")
    return match.group(1)


class SharedIsolationContractTests(unittest.TestCase):
    def test_isolation_and_clean_source_authentication_precede_first_python(self) -> None:
        for path in (SMOKE, PRODUCTION):
            with self.subTest(launcher=path.name):
                text = launcher_text(path)
                self.assertEqual(text.count("#SBATCH --no-requeue"), 1)
                lines = text.splitlines()
                python_calls = [
                    (index, line.strip())
                    for index, line in enumerate(lines)
                    if re.match(r'^"\$PYTHON_BIN"\s+-', line.strip())
                ]
                self.assertTrue(python_calls)
                first_python_line, first_python = python_calls[0]
                self.assertIn('"$PROJECT_DIR" "$STATE_SOURCE_DIR"', first_python)
                no_bytecode_line = lines.index("export PYTHONDONTWRITEBYTECODE=1")
                pythonpath_line = next(
                    index
                    for index, line in enumerate(lines)
                    if line.startswith('export PYTHONPATH="$STATE_SOURCE_DIR/src')
                )
                self.assertLess(no_bytecode_line, first_python_line)
                self.assertLess(pythonpath_line, first_python_line)
                self.assertIn(
                    "from public_candidate_v6_contract import authenticate_state_source",
                    text,
                )
                self.assertIn("require_active_isolation=True", text)
                self.assertIn(
                    f'readonly STATE_SOURCE_COMMIT="{EXPECTED_STATE_COMMIT}"', text
                )
                preflight = text.index("authentication = authenticate_state_source(")
                self.assertLess(preflight, text.index('mkdir -p -- "$RUN_ROOT"'))
                self.assertLess(preflight, text.index("import torch"))
                self.assertIn(
                    '"state_source_authentication": state_source_authentication',
                    text,
                )
                self.assertIn(
                    'state_source_authentication.get("repository_commit")', text
                )
                self.assertIn(
                    'state_source_authentication.get("tracked_worktree_clean") is not True',
                    text,
                )
                self.assertIn(
                    'state_source_authentication.get('
                    '"untracked_and_ignored_runtime_files_absent") is not True',
                    text,
                )

    def test_both_launchers_train_through_archive_shim(self) -> None:
        for path in (SMOKE, PRODUCTION):
            with self.subTest(launcher=path.name):
                text = launcher_text(path)
                self.assertEqual(text.count("#SBATCH --no-requeue"), 1)
                self.assertIn(
                    '"$PYTHON_BIN" "$CHECKPOINT_SHIM" "${STATE_ARGS[@]}"', text
                )
                self.assertNotIn('"$STATE_BIN" "${STATE_ARGS[@]}"', text)
                self.assertIn(
                    '"callback_state_schema": '
                    '"vcc-state-validation-step-archive-v1"',
                    text,
                )
                self.assertIn(
                    'receipt.get("checkpoint_layout") != "validation_step_archives"',
                    text,
                )
                self.assertIn('receipt.get("unarchived_validation_steps") != []', text)


class SmokeV1ContractTests(unittest.TestCase):
    def test_smoke_has_distinct_lineage_and_registered_twenty_step_archive(self) -> None:
        smoke = launcher_text(SMOKE)
        production = launcher_text(PRODUCTION)
        self.assertEqual(directive(smoke, "job-name"), "vcc-state-anvil-smoke-v1")
        self.assertEqual(
            directive(smoke, "output"), "logs/anvil_h100_state_smoke_v1_%j.out"
        )
        self.assertEqual(
            directive(smoke, "error"), "logs/anvil_h100_state_smoke_v1_%j.err"
        )
        self.assertNotIn("#SBATCH --array=", smoke)
        self.assertEqual(assignment(smoke, "RUN_NAME"), "state_sm_anvil_smoke_seed${TRAIN_SEED}_v1")
        self.assertNotEqual(assignment(smoke, "RUN_NAME"), assignment(production, "RUN_NAME"))
        self.assertIn('"schema": "vcc-anvil-state-training-smoke-runtime-v2"', smoke)
        self.assertIn(
            '"lineage": "anvil-state-training-smoke-clean-source-archive-v1"',
            smoke,
        )
        self.assertNotIn('"schema": "vcc-anvil-state-training-smoke-runtime-v2"', production)
        self.assertIn("training.max_steps=20", smoke)
        self.assertIn("training.val_freq=20", smoke)
        self.assertIn("training.ckpt_every_n_steps=20", smoke)
        self.assertIn('test -s "$RUN_DIR/checkpoints/step00020.ckpt"', smoke)
        self.assertIn("expected_steps = [20]", smoke)
        self.assertIn('get("global_step") != 20', smoke)
        self.assertIn('name != "step00020.ckpt"', smoke)
        self.assertIn('len(receipt.get("archived_candidates", [])) != 1', smoke)

    def test_production_retains_registered_array_and_runtime_schema(self) -> None:
        production = launcher_text(PRODUCTION)
        self.assertEqual(directive(production, "array"), "0-3%4")
        self.assertIn("training.max_steps=20000", production)
        self.assertIn("training.val_freq=2000", production)
        self.assertIn("training.ckpt_every_n_steps=2000", production)
        self.assertIn("expected_steps = list(range(2_000, 20_001, 2_000))", production)
        self.assertIn('"schema": "vcc-anvil-state-training-runtime-v2"', production)
        self.assertIn(
            '"lineage": "anvil-state-training-production-clean-source-archives-v1"',
            production,
        )


if __name__ == "__main__":
    unittest.main()
