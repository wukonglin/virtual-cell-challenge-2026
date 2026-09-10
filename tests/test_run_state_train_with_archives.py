"""Focused synthetic tests for the STATE validation archive shim."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from lightning.pytorch.callbacks import ModelCheckpoint


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_state_train_with_archives as archive  # noqa: E402


class FakeTrainer:
    def __init__(
        self,
        callback: archive.VCCValidationStepArchive,
        *,
        step: int = 20,
        val_loss: object = 0.25,
        sanity_checking: bool = False,
    ) -> None:
        self.callback = callback
        self.global_step = step
        self.callback_metrics = {"val_loss": val_loss}
        self.sanity_checking = sanity_checking
        self.world_size = 1
        self.is_global_zero = True
        self.save_calls: list[tuple[str, bool]] = []

    def save_checkpoint(self, path: str, weights_only: bool = False) -> None:
        self.save_calls.append((path, weights_only))
        torch.save(
            {
                "global_step": self.global_step,
                "callbacks": {self.callback.state_key: self.callback.state_dict()},
                "state_dict": {"weight": torch.arange(3, dtype=torch.float32)},
            },
            path,
        )




def checkpoint_callback(directory: Path, cadence: int = 20) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=directory,
        filename="best",
        save_last=True,
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_train_steps=cadence,
    )

class ValidationStepArchiveTests(unittest.TestCase):
    def test_saves_regular_atomic_archive_with_current_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            callback = archive.VCCValidationStepArchive(checkpoint_dir)
            trainer = FakeTrainer(callback, step=2_000, val_loss=torch.tensor(0.125))

            callback.on_validation_end(trainer, object())

            result = checkpoint_dir / "step02000.ckpt"
            self.assertTrue(result.is_file())
            self.assertFalse(result.is_symlink())
            self.assertTrue(stat.S_ISREG(result.lstat().st_mode))
            self.assertEqual(len(trainer.save_calls), 1)
            self.assertFalse(trainer.save_calls[0][1])
            self.assertFalse(any(path.name.endswith(".tmp") for path in checkpoint_dir.iterdir()))
            payload = torch.load(result, map_location="cpu", weights_only=True)
            self.assertEqual(payload["global_step"], 2_000)
            state = payload["callbacks"][archive.ARCHIVE_STATE_KEY]
            self.assertEqual(state["schema"], archive.ARCHIVE_SCHEMA)
            self.assertEqual(state["global_step"], 2_000)
            self.assertEqual(state["val_loss"], 0.125)
            self.assertEqual(state["filename"], "step02000.ckpt")
            self.assertEqual(state["archive_count"], 1)

    def test_sanity_validation_does_not_save(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            callback = archive.VCCValidationStepArchive(checkpoint_dir)
            trainer = FakeTrainer(callback, sanity_checking=True)
            callback.on_validation_end(trainer, object())
            self.assertEqual(trainer.save_calls, [])
            self.assertFalse(checkpoint_dir.exists())

    def test_missing_or_nonfinite_val_loss_fails_closed(self) -> None:
        cases = (float("nan"), float("inf"), [0.2, 0.3])
        for value in cases:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                callback = archive.VCCValidationStepArchive(Path(temporary) / "checkpoints")
                trainer = FakeTrainer(callback, val_loss=value)
                with self.assertRaises(archive.ArchiveContractError):
                    callback.on_validation_end(trainer, object())
                self.assertEqual(trainer.save_calls, [])

        with tempfile.TemporaryDirectory() as temporary:
            callback = archive.VCCValidationStepArchive(Path(temporary) / "checkpoints")
            trainer = FakeTrainer(callback)
            trainer.callback_metrics = {}
            with self.assertRaisesRegex(archive.ArchiveContractError, "without val_loss"):
                callback.on_validation_end(trainer, object())

    def test_existing_destination_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            checkpoint_dir.mkdir()
            destination = checkpoint_dir / "step00020.ckpt"
            destination.write_bytes(b"incumbent")
            callback = archive.VCCValidationStepArchive(checkpoint_dir)
            trainer = FakeTrainer(callback)
            with self.assertRaisesRegex(archive.ArchiveContractError, "Refusing to overwrite"):
                callback.on_validation_end(trainer, object())
            self.assertEqual(destination.read_bytes(), b"incumbent")
            self.assertEqual(trainer.save_calls, [])

    def test_save_failure_cleans_temp_and_rolls_back_callback_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_dir = Path(temporary) / "checkpoints"
            callback = archive.VCCValidationStepArchive(checkpoint_dir)
            trainer = FakeTrainer(callback)

            def fail(path: str, weights_only: bool = False) -> None:
                del weights_only
                Path(path).write_bytes(b"partial")
                raise RuntimeError("synthetic save failure")

            trainer.save_checkpoint = fail  # type: ignore[method-assign]
            with self.assertRaisesRegex(RuntimeError, "synthetic save failure"):
                callback.on_validation_end(trainer, object())
            self.assertEqual(list(checkpoint_dir.iterdir()), [])
            self.assertIsNone(callback.state_dict()["global_step"])
            self.assertEqual(callback.state_dict()["archive_count"], 0)

    def test_original_best_last_callback_is_preserved_by_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary) / "runs"
            name = "fit"
            original = ModelCheckpoint(
                dirpath=output_dir / name / "checkpoints",
                filename="best",
                save_last=True,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                every_n_train_steps=20,
            )

            def factory(*args: object) -> list[ModelCheckpoint]:
                del args
                return [original]

            patched = archive.make_patched_checkpoint_factory(factory)
            callbacks = patched(str(output_dir), name, 20, 20)
            self.assertIs(callbacks[0], original)
            self.assertIsInstance(callbacks[1], archive.VCCValidationStepArchive)
            self.assertEqual(len(callbacks), 2)
            self.assertTrue(callbacks[0].save_last)

    def test_accepts_symlinked_parent_alias_and_uses_canonical_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_output = root / "real" / "runs"
            checkpoint_dir = real_output / "fit" / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            alias_output = root / "alias"
            os.symlink(real_output, alias_output)
            original = checkpoint_callback(checkpoint_dir)

            def factory(*args: object) -> list[ModelCheckpoint]:
                del args
                return [original]

            callbacks = archive.make_patched_checkpoint_factory(factory)(
                str(alias_output), "fit", 20, 20
            )
            self.assertIs(callbacks[0], original)
            self.assertEqual(callbacks[1].checkpoint_dir, checkpoint_dir.resolve())

    def test_rejects_different_canonical_parent_even_with_exact_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected_output = root / "expected"
            observed = root / "other" / "fit" / "checkpoints"
            original = checkpoint_callback(observed)

            def factory(*args: object) -> list[ModelCheckpoint]:
                del args
                return [original]

            patched = archive.make_patched_checkpoint_factory(factory)
            with self.assertRaisesRegex(archive.ArchiveContractError, "directory changed"):
                patched(str(expected_output), "fit", 20, 20)

    def test_rejects_changed_or_symlink_checkpoint_leaf(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(archive.ArchiveContractError, "leaf must be exactly"):
                archive._canonical_checkpoint_directory(root / "not-checkpoints", "test")

            target = root / "real-checkpoints"
            target.mkdir()
            leaf = root / "checkpoints"
            os.symlink(target, leaf)
            with self.assertRaisesRegex(archive.ArchiveContractError, "must not be a symlink"):
                archive._canonical_checkpoint_directory(leaf, "test")

    def test_symlink_checkpoint_directory_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real_dir = root / "real"
            real_dir.mkdir()
            link = root / "checkpoints"
            os.symlink(real_dir, link)
            callback = archive.VCCValidationStepArchive(link)
            trainer = FakeTrainer(callback)
            with self.assertRaisesRegex(archive.ArchiveContractError, "must not be a symlink"):
                callback.on_validation_end(trainer, object())
            self.assertEqual(trainer.save_calls, [])


if __name__ == "__main__":
    unittest.main()
