"""Focused synthetic tests for STATE checkpoint selection layouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import select_state_checkpoint as selector  # noqa: E402


def make_run(root: Path, rows: list[dict[str, object]]) -> Path:
    run_dir = root / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "version_0").mkdir()
    pd.DataFrame(rows).to_csv(run_dir / "version_0" / "metrics.csv", index=False)
    return run_dir


def args_for(run_dir: Path) -> argparse.Namespace:
    return argparse.Namespace(run_dir=run_dir, metrics=None, output_link=None, output_json=None)


def write_checkpoint(
    path: Path,
    global_step: int,
    *,
    best_score: float | None = None,
    best_path: str = "/old/run/checkpoints/best.ckpt",
    archive_loss: float | None = None,
    archive_count: int = 1,
    archive_filename: str | None = None,
    archive_overrides: dict[str, object] | None = None,
) -> None:
    callbacks: dict[str, object] = {}
    if best_score is not None:
        callbacks["ModelCheckpoint{'monitor': 'val_loss', 'mode': 'min'}"] = {
            "best_model_score": torch.tensor(best_score),
            "best_model_path": best_path,
        }
    if archive_loss is not None:
        archive_state: dict[str, object] = {
            "schema": selector.ARCHIVE_CALLBACK_SCHEMA,
            "global_step": global_step,
            "val_loss": archive_loss,
            "filename": archive_filename or f"step{global_step:05d}.ckpt",
            "archive_count": archive_count,
            "hook": selector.ARCHIVE_CALLBACK_HOOK,
            "publish_method": selector.ARCHIVE_PUBLISH_METHOD,
        }
        archive_state.update(archive_overrides or {})
        callbacks[selector.ARCHIVE_CALLBACK_STATE_KEY] = archive_state
    torch.save(
        {
            "global_step": global_step,
            "state_dict": {"weight": torch.arange(4, dtype=torch.float32)},
            "callbacks": callbacks,
        },
        path,
    )


class StepArchiveLayoutTests(unittest.TestCase):
    def test_selects_minimum_among_complete_authenticated_archives(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(
                Path(temporary),
                [
                    {"step": 9, "val_loss": 0.5},
                    {"step": 19, "val_loss": 0.2},
                    {"step": 29, "val_loss": 0.3},
                ],
            )
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(
                checkpoint_dir / "step00010.ckpt", 10, archive_loss=0.5, archive_count=1
            )
            write_checkpoint(
                checkpoint_dir / "step00020.ckpt", 20, archive_loss=0.2, archive_count=2
            )
            write_checkpoint(
                checkpoint_dir / "step00030.ckpt", 30, archive_loss=0.3, archive_count=3
            )

            payload = selector.run(args_for(run_dir))
            selected = checkpoint_dir / "selected.ckpt"
            receipt = checkpoint_dir / "selected_checkpoint.json"
            self.assertEqual(payload["schema"], selector.SCHEMA)
            self.assertEqual(payload["checkpoint_layout"], selector.STEP_ARCHIVE_LAYOUT)
            self.assertEqual(payload["selected"]["global_step"], 20)
            self.assertEqual(payload["selected"]["val_loss"], 0.2)
            self.assertEqual(payload["unarchived_validation_steps"], [])
            self.assertTrue(
                payload["layout_authentication"]["all_finite_validation_steps_archived"]
            )
            self.assertEqual(
                len(payload["layout_authentication"]["authenticated_archives"]), 3
            )
            self.assertTrue(selected.is_symlink())
            self.assertEqual(os.readlink(selected), "step00020.ckpt")
            self.assertEqual(json.loads(receipt.read_text()), payload)

    def test_existing_outputs_are_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(Path(temporary), [{"step": 9, "val_loss": 0.5}])
            checkpoint = run_dir / "checkpoints" / "step00010.ckpt"
            write_checkpoint(checkpoint, 10, archive_loss=0.5)
            first = selector.run(args_for(run_dir))
            receipt_path = run_dir / "checkpoints" / "selected_checkpoint.json"
            original_receipt = receipt_path.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                selector.run(args_for(run_dir))
            self.assertEqual(receipt_path.read_bytes(), original_receipt)
            self.assertEqual(json.loads(original_receipt), first)

    def test_partial_archive_layout_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(
                Path(temporary),
                [
                    {"step": 9, "val_loss": 0.5},
                    {"step": 19, "val_loss": 0.2},
                    {"step": 29, "val_loss": 0.3},
                ],
            )
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(
                checkpoint_dir / "step00010.ckpt", 10, archive_loss=0.5, archive_count=1
            )
            write_checkpoint(
                checkpoint_dir / "step00030.ckpt", 30, archive_loss=0.3, archive_count=3
            )
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "incomplete"):
                selector.run(args_for(run_dir))
            self.assertFalse((checkpoint_dir / "selected.ckpt").exists())

    def test_archive_root_and_callback_metadata_are_authenticated(self) -> None:
        mutations = {
            "root global_step": {"root_step": 11},
            "callback global_step": {"overrides": {"global_step": 11}},
            "callback filename": {"filename": "step99999.ckpt"},
            "callback val_loss": {"loss": 0.6},
            "archive_count": {"count": 2},
            "not an integer": {"overrides": {"archive_count": 1.0}},
            "filename is missing": {"overrides": {"filename": None}},
            "callback schema": {"overrides": {"schema": "wrong"}},
        }
        for expected_error, mutation in mutations.items():
            with self.subTest(expected_error=expected_error), tempfile.TemporaryDirectory() as temporary:
                run_dir = make_run(Path(temporary), [{"step": 9, "val_loss": 0.5}])
                checkpoint_dir = run_dir / "checkpoints"
                write_checkpoint(
                    checkpoint_dir / "step00010.ckpt",
                    int(mutation.get("root_step", 10)),
                    archive_loss=float(mutation.get("loss", 0.5)),
                    archive_count=int(mutation.get("count", 1)),
                    archive_filename=mutation.get("filename"),
                    archive_overrides=mutation.get("overrides"),
                )
                with self.assertRaisesRegex(selector.CheckpointSelectionError, expected_error):
                    selector.run(args_for(run_dir))

    def test_archive_without_route_callback_state_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(Path(temporary), [{"step": 9, "val_loss": 0.5}])
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(checkpoint_dir / "step00010.ckpt", 10)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "exactly one"):
                selector.run(args_for(run_dir))


class BestLastLayoutTests(unittest.TestCase):
    @staticmethod
    def _standard_run(root: Path) -> Path:
        run_dir = make_run(
            root,
            [
                {"step": 9, "val_loss": 0.5},
                {"step": 19, "val_loss": 0.2},
                {"step": 29, "val_loss": 0.3},
            ],
        )
        write_checkpoint(run_dir / "checkpoints" / "best.ckpt", 20, best_score=0.2)
        write_checkpoint(run_dir / "checkpoints" / "last.ckpt", 30, best_score=0.2)
        return run_dir

    def test_authenticates_best_and_materializes_regular_step_hardlink(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._standard_run(Path(temporary))
            checkpoint_dir = run_dir / "checkpoints"
            best = checkpoint_dir / "best.ckpt"

            payload = selector.run(args_for(run_dir))
            archive = checkpoint_dir / "step00020.ckpt"
            selected = checkpoint_dir / "selected.ckpt"
            receipt = checkpoint_dir / "selected_checkpoint.json"
            self.assertEqual(payload["checkpoint_layout"], selector.BEST_LAST_LAYOUT)
            self.assertTrue(archive.is_file())
            self.assertFalse(archive.is_symlink())
            self.assertTrue(stat.S_ISREG(archive.stat().st_mode))
            self.assertEqual((best.stat().st_dev, best.stat().st_ino), (archive.stat().st_dev, archive.stat().st_ino))
            self.assertEqual(os.readlink(selected), "step00020.ckpt")
            self.assertEqual(Path(payload["selected"]["checkpoint"]), archive)
            self.assertEqual(
                payload["selected"]["checkpoint_sha256"],
                hashlib.sha256(best.read_bytes()).hexdigest(),
            )
            authentication = payload["layout_authentication"]
            self.assertTrue(authentication["best_global_step_matches_minimum"])
            self.assertTrue(authentication["callback_best_score_available"])
            self.assertEqual(authentication["selected_source"], "best.ckpt")
            self.assertEqual(len(authentication["callback_best_score_checks"]), 1)
            self.assertTrue(authentication["materialization"]["materialized"])
            self.assertTrue(authentication["materialization"]["same_device_and_inode"])
            self.assertEqual(json.loads(receipt.read_text()), payload)

    def test_fallback_requires_independent_loss_corroboration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(Path(temporary), [{"step": 19, "val_loss": 0.2}])
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(checkpoint_dir / "best.ckpt", 20, best_score=None)
            write_checkpoint(checkpoint_dir / "last.ckpt", 20, best_score=None)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "defensible cross-check"):
                selector.run(args_for(run_dir))

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(Path(temporary), [{"step": 19, "val_loss": 0.2}])
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(checkpoint_dir / "best.ckpt", 20, best_score=0.9)
            write_checkpoint(checkpoint_dir / "last.ckpt", 20, best_score=0.8)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "defensible cross-check"):
                selector.run(args_for(run_dir))
            self.assertFalse((checkpoint_dir / "step00020.ckpt").exists())

    def test_stale_best_is_rejected_and_verified_last_is_materialized(self) -> None:
        """Reproduce the observed step-40 stale best-score ordering defect."""

        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(
                Path(temporary),
                [
                    {"step": 19, "val_loss": 20.491},
                    {"step": 39, "val_loss": 15.510},
                ],
            )
            checkpoint_dir = run_dir / "checkpoints"
            best = checkpoint_dir / "best.ckpt"
            last = checkpoint_dir / "last.ckpt"
            write_checkpoint(best, 40, best_score=20.491)
            write_checkpoint(last, 40, best_score=15.510)

            payload = selector.run(args_for(run_dir))
            archive = checkpoint_dir / "step00040.ckpt"
            authentication = payload["layout_authentication"]
            self.assertEqual(authentication["selected_source"], "last.ckpt")
            self.assertFalse(authentication["source_eligibility_checks"][0]["eligible_source"])
            self.assertTrue(authentication["source_eligibility_checks"][1]["eligible_source"])
            self.assertEqual(
                (last.stat().st_dev, last.stat().st_ino),
                (archive.stat().st_dev, archive.stat().st_ino),
            )
            self.assertNotEqual(best.stat().st_ino, archive.stat().st_ino)
            self.assertEqual(
                payload["selected"]["checkpoint_sha256"],
                hashlib.sha256(last.read_bytes()).hexdigest(),
            )

    def test_route_archive_state_can_corroborate_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(Path(temporary), [{"step": 19, "val_loss": 0.2}])
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(
                checkpoint_dir / "best.ckpt",
                20,
                archive_loss=0.2,
                archive_count=1,
            )
            write_checkpoint(checkpoint_dir / "last.ckpt", 20)
            payload = selector.run(args_for(run_dir))
            authentication = payload["layout_authentication"]
            self.assertEqual(authentication["selected_source"], "best.ckpt")
            self.assertFalse(authentication["callback_best_score_available"])
            self.assertTrue(
                authentication["selected_source_check"][
                    "validation_archive_callback_crosscheck"
                ]
            )

    def test_best_global_step_must_equal_earliest_minimum_finite_row(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._standard_run(Path(temporary))
            checkpoint_dir = run_dir / "checkpoints"
            (checkpoint_dir / "best.ckpt").unlink()
            write_checkpoint(checkpoint_dir / "best.ckpt", 10, best_score=0.2)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "defensible cross-check"):
                selector.run(args_for(run_dir))
            self.assertFalse((checkpoint_dir / "step00020.ckpt").exists())
            self.assertFalse((checkpoint_dir / "selected.ckpt").exists())

    def test_last_checkpoint_cannot_predate_best(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = self._standard_run(Path(temporary))
            checkpoint_dir = run_dir / "checkpoints"
            (checkpoint_dir / "last.ckpt").unlink()
            write_checkpoint(checkpoint_dir / "last.ckpt", 10, best_score=0.2)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "predates"):
                selector.run(args_for(run_dir))

    def test_nonfinite_losses_are_excluded_from_minimum(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(
                Path(temporary),
                [{"step": 9, "val_loss": float("inf")}, {"step": 19, "val_loss": 0.2}],
            )
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(checkpoint_dir / "best.ckpt", 20, best_score=0.2)
            write_checkpoint(checkpoint_dir / "last.ckpt", 20, best_score=0.2)
            payload = selector.run(args_for(run_dir))
            self.assertEqual(payload["validation_results"], [{"global_step": 20, "val_loss": 0.2}])
            self.assertEqual(
                payload["nonfinite_validation_results_ignored"],
                [{"global_step": 10, "val_loss": "inf"}],
            )

    def test_fallback_requires_exact_best_plus_last_layout(self) -> None:
        for mutation in ("missing_last", "extra_final"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                run_dir = make_run(Path(temporary), [{"step": 19, "val_loss": 0.2}])
                checkpoint_dir = run_dir / "checkpoints"
                write_checkpoint(checkpoint_dir / "best.ckpt", 20, best_score=0.2)
                if mutation != "missing_last":
                    write_checkpoint(checkpoint_dir / "last.ckpt", 20, best_score=0.2)
                if mutation == "extra_final":
                    write_checkpoint(checkpoint_dir / "final.ckpt", 20, best_score=0.2)
                with self.assertRaisesRegex(selector.CheckpointSelectionError, "exactly"):
                    selector.run(args_for(run_dir))

    def test_duplicate_finite_metric_step_is_ambiguous(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = make_run(
                Path(temporary),
                [{"step": 19, "val_loss": 0.2}, {"step": 19, "val_loss": 0.3}],
            )
            checkpoint_dir = run_dir / "checkpoints"
            write_checkpoint(checkpoint_dir / "best.ckpt", 20, best_score=0.2)
            write_checkpoint(checkpoint_dir / "last.ckpt", 20, best_score=0.2)
            with self.assertRaisesRegex(selector.CheckpointSelectionError, "duplicate"):
                selector.run(args_for(run_dir))


if __name__ == "__main__":
    unittest.main()
