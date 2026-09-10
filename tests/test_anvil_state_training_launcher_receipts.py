"""Static syntax and provenance checks for Anvil STATE training receipts."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
LAUNCHERS = (
    REPOSITORY / "slurm" / "anvil_h100_train_state_smoke_v1.sbatch",
    REPOSITORY / "slurm" / "anvil_h100_train_state_sm_20k_array_v1.sbatch",
)
EXPECTED_FEATURES_SHA256 = (
    "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748"
)
SUPPORT_FILES = (
    "competition_train.h5",
    "k562_gwps.h5",
    "rpe1.h5",
    "jurkat.h5",
    "k562.h5",
    "hepg2.h5",
)


class LauncherReceiptTests(unittest.TestCase):
    def test_every_embedded_python_program_compiles(self) -> None:
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                text = launcher.read_text(encoding="utf-8")
                programs = re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.DOTALL)
                self.assertGreaterEqual(len(programs), 5)
                for index, program in enumerate(programs):
                    with self.subTest(program=index):
                        compile(program, f"{launcher.name}:heredoc-{index}", "exec")

    def test_runtime_receipt_binds_training_inputs_and_source_tree(self) -> None:
        for launcher in LAUNCHERS:
            with self.subTest(launcher=launcher.name):
                text = launcher.read_text(encoding="utf-8")
                self.assertIn(
                    '"state_source_authentication": state_source_authentication',
                    text,
                )
                self.assertIn('"training_inputs": training_inputs', text)
                self.assertIn('"state_config": config_source', text)
                self.assertIn(
                    f'EXPECTED_FEATURES_SHA256 = "{EXPECTED_FEATURES_SHA256}"',
                    text,
                )
                self.assertIn('"registered_sha256_match": True', text)
                self.assertIn('"registered_size_match": True', text)
                self.assertIn('"state_support_wrappers": support_inputs', text)
                self.assertIn('"linked_payload": regular_stat_identity(', text)
                self.assertIn(
                    '"linked_payload_cryptographic_hash_computed_per_task": False',
                    text,
                )
                for filename in SUPPORT_FILES:
                    self.assertIn(f'"{filename}"', text)


if __name__ == "__main__":
    unittest.main()
