#!/usr/bin/env python3
"""Run pinned STATE training with authenticated validation-step archives.

The pinned STATE callback factory writes ``best.ckpt`` and ``last.ckpt`` but
does not retain a checkpoint for every validation result.  This wrapper leaves
that factory's original ``ModelCheckpoint`` object in place and appends one
callback that publishes ``step{global_step:05d}.ckpt`` after every real
validation loop.  Publication is a same-directory hard link from a fully
written temporary checkpoint, so an existing archive is never replaced.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import math
import os
import secrets
import stat
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from lightning.pytorch.callbacks import Callback, Checkpoint, ModelCheckpoint


ARCHIVE_SCHEMA = "vcc-state-validation-step-archive-v1"
ARCHIVE_STATE_KEY = "VCCValidationStepArchive"
EXPECTED_ARC_STATE_VERSION = "0.11.3"
EXPECTED_LIGHTNING_VERSION = "2.5.2"
EXPECTED_STATE_UTILS_SHA256 = (
    "6937a86c5fe8c091d064fd58fdf81645604a64277cea0f46eef4d3b829d8b6a2"
)
EXPECTED_FACTORY_PARAMETERS = (
    "output_dir",
    "name",
    "val_freq",
    "_ckpt_every_n_steps",
)


class ArchiveContractError(RuntimeError):
    """The pinned callback or archive contract could not be satisfied."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ArchiveContractError(message)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def _scalar_float(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ArchiveContractError(f"{label} must be numeric, not boolean")
    if hasattr(value, "detach") and hasattr(value, "numel"):
        require(int(value.numel()) == 1, f"{label} must be scalar")
        value = value.detach().cpu().item()
    elif hasattr(value, "size") and not isinstance(value, (str, bytes)):
        size = value.size
        size = size() if callable(size) else size
        if isinstance(size, int):
            require(size == 1, f"{label} must be scalar")
        if hasattr(value, "item"):
            value = value.item()
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArchiveContractError(f"{label} must be numeric") from error
    require(math.isfinite(result), f"{label} must be finite")
    return result


def _positive_global_step(value: Any) -> int:
    if isinstance(value, bool):
        raise ArchiveContractError("trainer.global_step must be a positive integer")
    try:
        step = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ArchiveContractError(
            "trainer.global_step must be a positive integer"
        ) from error
    require(step > 0, "trainer.global_step must be positive")
    try:
        exact = bool(value == step)
    except Exception as error:
        raise ArchiveContractError("trainer.global_step is not integral") from error
    require(exact, "trainer.global_step is not integral")
    return step


def _canonical_checkpoint_directory(path: str | os.PathLike[str], label: str) -> Path:
    lexical = Path(os.path.abspath(os.fspath(Path(path).expanduser())))
    require(lexical.name == "checkpoints", f"{label} leaf must be exactly checkpoints")
    if os.path.lexists(lexical):
        metadata = lexical.lstat()
        require(not lexical.is_symlink(), f"{label} leaf must not be a symlink: {lexical}")
        require(stat.S_ISDIR(metadata.st_mode), f"{label} is not a directory: {lexical}")
    try:
        # strict=False resolves every existing parent alias while retaining an
        # as-yet-uncreated checkpoints leaf. Parent aliases such as /fs and
        # /NFS4 are valid only when they identify the same canonical target.
        canonical = lexical.resolve(strict=False)
    except (OSError, RuntimeError) as error:
        raise ArchiveContractError(f"Cannot canonicalize {label}: {lexical}") from error
    require(canonical.name == "checkpoints", f"{label} canonical leaf changed")
    return canonical


def _prepare_regular_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata = path.lstat()
    require(not path.is_symlink(), f"Checkpoint directory must not be a symlink: {path}")
    require(stat.S_ISDIR(metadata.st_mode), f"Checkpoint path is not a directory: {path}")


class VCCValidationStepArchive(Checkpoint):
    """Save one immutable, metadata-bearing checkpoint per validation loop."""

    def __init__(self, checkpoint_dir: str | os.PathLike[str]) -> None:
        super().__init__()
        self.checkpoint_dir = Path(
            os.path.abspath(os.fspath(Path(checkpoint_dir).expanduser()))
        )
        self._global_step: int | None = None
        self._val_loss: float | None = None
        self._filename: str | None = None
        self._archive_count = 0

    @property
    def state_key(self) -> str:
        return ARCHIVE_STATE_KEY

    def state_dict(self) -> dict[str, Any]:
        # This state is also serialized into best/last before the first archive,
        # where the three current-archive fields are intentionally None.
        return {
            "schema": ARCHIVE_SCHEMA,
            "global_step": self._global_step,
            "val_loss": self._val_loss,
            "filename": self._filename,
            "archive_count": self._archive_count,
            "hook": "on_validation_end",
            "publish_method": (
                "same_directory_temporary_then_atomic_hard_link_no_overwrite"
            ),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        require(state_dict.get("schema") == ARCHIVE_SCHEMA, "Archive state schema mismatch")
        count = state_dict.get("archive_count")
        require(isinstance(count, int) and count >= 0, "Archive count is invalid")
        self._archive_count = count
        self._global_step = state_dict.get("global_step")
        self._val_loss = state_dict.get("val_loss")
        self._filename = state_dict.get("filename")

    def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
        del pl_module
        if bool(getattr(trainer, "sanity_checking", False)):
            return
        require(
            int(getattr(trainer, "world_size", 1)) == 1,
            "Validation-step archiving is registered only for one-process training",
        )
        require(bool(getattr(trainer, "is_global_zero", False)), "Trainer is not global rank zero")

        metrics = getattr(trainer, "callback_metrics", None)
        require(isinstance(metrics, Mapping), "trainer.callback_metrics is not a mapping")
        require("val_loss" in metrics, "Validation ended without val_loss")
        loss = _scalar_float(metrics["val_loss"], "val_loss")
        step = _positive_global_step(getattr(trainer, "global_step", None))
        filename = f"step{step:05d}.ckpt"

        _prepare_regular_directory(self.checkpoint_dir)
        destination = self.checkpoint_dir / filename
        require(
            not os.path.lexists(destination),
            f"Refusing to overwrite validation-step archive: {destination}",
        )
        temporary = self.checkpoint_dir / (
            f".{filename}.{os.getpid()}.{secrets.token_hex(12)}.tmp"
        )
        require(not os.path.lexists(temporary), f"Temporary checkpoint already exists: {temporary}")

        previous_state = (
            self._global_step,
            self._val_loss,
            self._filename,
            self._archive_count,
        )
        self._global_step = step
        self._val_loss = loss
        self._filename = filename
        self._archive_count += 1
        published = False
        try:
            trainer.save_checkpoint(str(temporary), weights_only=False)
            temporary_stat = temporary.lstat()
            require(
                stat.S_ISREG(temporary_stat.st_mode) and not temporary.is_symlink(),
                "Trainer did not create a regular temporary checkpoint",
            )
            require(temporary_stat.st_size > 0, "Trainer created an empty checkpoint")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary, flags)
            try:
                opened_stat = os.fstat(descriptor)
                require(
                    (opened_stat.st_dev, opened_stat.st_ino, opened_stat.st_size)
                    == (temporary_stat.st_dev, temporary_stat.st_ino, temporary_stat.st_size),
                    "Temporary checkpoint changed before publication",
                )
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            try:
                os.link(temporary, destination, follow_symlinks=False)
            except FileExistsError as error:
                raise ArchiveContractError(
                    f"Refusing to overwrite validation-step archive: {destination}"
                ) from error
            published = True
            destination_stat = destination.lstat()
            require(
                stat.S_ISREG(destination_stat.st_mode) and not destination.is_symlink(),
                "Published validation-step archive is not a regular file",
            )
            require(
                (destination_stat.st_dev, destination_stat.st_ino, destination_stat.st_size)
                == (temporary_stat.st_dev, temporary_stat.st_ino, temporary_stat.st_size),
                "Published archive is not the authenticated temporary checkpoint",
            )
        except Exception:
            if not published:
                (
                    self._global_step,
                    self._val_loss,
                    self._filename,
                    self._archive_count,
                ) = previous_state
            raise
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


def make_patched_checkpoint_factory(
    original_factory: Callable[..., list[Callback]],
) -> Callable[..., list[Callback]]:
    """Return a strict wrapper that preserves STATE's original callback object."""

    def patched(
        output_dir: str,
        name: str,
        val_freq: int,
        _ckpt_every_n_steps: int,
    ) -> list[Callback]:
        require(int(val_freq) > 0, "val_freq must be positive")
        require(int(_ckpt_every_n_steps) > 0, "ckpt_every_n_steps must be positive")
        require(
            int(_ckpt_every_n_steps) == int(val_freq),
            "Registered archive cadence requires ckpt_every_n_steps == val_freq",
        )
        callbacks = original_factory(output_dir, name, val_freq, _ckpt_every_n_steps)
        require(type(callbacks) is list, "Pinned checkpoint factory did not return a list")
        require(len(callbacks) == 1, "Pinned checkpoint factory callback count changed")
        original = callbacks[0]
        require(
            isinstance(original, ModelCheckpoint),
            "Pinned checkpoint factory no longer returns ModelCheckpoint",
        )
        require(original.filename == "best", "Pinned checkpoint filename changed")
        require(original.save_last is True, "Pinned last.ckpt behavior changed")
        require(original.monitor == "val_loss", "Pinned checkpoint monitor changed")
        require(original.mode == "min", "Pinned checkpoint mode changed")
        require(original.save_top_k == 1, "Pinned checkpoint retention changed")
        require(
            original._every_n_train_steps == int(val_freq),
            "Pinned checkpoint validation cadence changed",
        )
        expected_dir = _canonical_checkpoint_directory(
            os.path.join(output_dir, name, "checkpoints"), "Expected checkpoint directory"
        )
        original_dir = _canonical_checkpoint_directory(
            os.fspath(original.dirpath), "Pinned checkpoint directory"
        )
        require(original_dir == expected_dir, "Pinned checkpoint directory changed")
        # Keep the exact original object first so its best.ckpt/last.ckpt behavior
        # is unchanged and serialized before the validation-aligned archive.
        return [*callbacks, VCCValidationStepArchive(expected_dir)]

    return patched


def install_checkpoint_patch() -> None:
    import state.tx.utils as state_utils

    lightning_version = importlib.metadata.version("lightning")
    require(
        lightning_version == EXPECTED_LIGHTNING_VERSION,
        f"Expected lightning {EXPECTED_LIGHTNING_VERSION}, found {lightning_version}",
    )
    version = importlib.metadata.version("arc-state")
    require(
        version == EXPECTED_ARC_STATE_VERSION,
        f"Expected arc-state {EXPECTED_ARC_STATE_VERSION}, found {version}",
    )
    source_path = Path(inspect.getsourcefile(state_utils) or "").resolve()
    expected_path = (
        Path(__file__).resolve().parents[1]
        / "external"
        / "state_runtime_9bbfe78a"
        / "src"
        / "state"
        / "tx"
        / "utils"
        / "__init__.py"
    ).resolve()
    require(source_path == expected_path, f"STATE utils resolved outside pinned checkout: {source_path}")
    require(
        sha256_file(source_path) == EXPECTED_STATE_UTILS_SHA256,
        "Pinned STATE utils source hash changed",
    )
    original = state_utils.get_checkpoint_callbacks
    require(
        tuple(inspect.signature(original).parameters) == EXPECTED_FACTORY_PARAMETERS,
        "Pinned checkpoint factory signature changed",
    )
    require(
        not bool(getattr(original, "_vcc_validation_archive_patch", False)),
        "Checkpoint callback factory is already patched",
    )
    patched = make_patched_checkpoint_factory(original)
    patched._vcc_validation_archive_patch = True  # type: ignore[attr-defined]
    state_utils.get_checkpoint_callbacks = patched


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    require(arguments[:2] == ["tx", "train"], "Wrapper accepts only `state tx train` arguments")
    install_checkpoint_patch()
    from state.__main__ import main as state_main

    if argv is None:
        state_main()
    else:
        previous = sys.argv
        try:
            sys.argv = [previous[0], *arguments]
            state_main()
        finally:
            sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
