"""Tiny real-Lightning integration test for validation archive timing."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import torch
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import run_state_train_with_archives as archive  # noqa: E402


class TinyValidationModel(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def training_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> torch.Tensor:
        del batch_idx
        return (self.weight * batch[0]).square().mean()

    def validation_step(self, batch: tuple[torch.Tensor], batch_idx: int) -> None:
        del batch, batch_idx
        current_loss = torch.tensor(float(10 - self.trainer.global_step), device=self.device)
        self.log("val_loss", current_loss, on_step=False, on_epoch=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.01)


class LightningArchiveTimingTests(unittest.TestCase):
    def test_archives_current_validation_not_previous_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint_dir = root / "run" / "checkpoints"
            original = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename="best",
                save_last=True,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                every_n_train_steps=2,
            )

            def factory(*args: object) -> list[ModelCheckpoint]:
                del args
                return [original]

            callbacks = archive.make_patched_checkpoint_factory(factory)(
                str(root), "run", 2, 2
            )
            trainer = Trainer(
                accelerator="cpu",
                devices=1,
                max_steps=4,
                check_val_every_n_epoch=None,
                val_check_interval=2,
                limit_val_batches=1,
                num_sanity_val_steps=0,
                callbacks=callbacks,
                logger=False,
                enable_progress_bar=False,
                enable_model_summary=False,
            )
            self.assertLess(
                trainer.callbacks.index(callbacks[0]),
                trainer.callbacks.index(callbacks[1]),
            )
            data = TensorDataset(torch.ones(8, 1))
            loader = DataLoader(data, batch_size=1)
            trainer.fit(TinyValidationModel(), train_dataloaders=loader, val_dataloaders=loader)

            for count, (step, expected_loss) in enumerate(((2, 8.0), (4, 6.0)), start=1):
                path = checkpoint_dir / f"step{step:05d}.ckpt"
                checkpoint = torch.load(path, map_location="cpu", weights_only=True)
                self.assertEqual(checkpoint["global_step"], step)
                state = checkpoint["callbacks"][archive.ARCHIVE_STATE_KEY]
                self.assertEqual(state["schema"], archive.ARCHIVE_SCHEMA)
                self.assertEqual(state["global_step"], step)
                self.assertEqual(state["val_loss"], expected_loss)
                self.assertEqual(state["filename"], path.name)
                self.assertEqual(state["archive_count"], count)


if __name__ == "__main__":
    unittest.main()
