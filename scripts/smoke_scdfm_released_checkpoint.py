#!/usr/bin/env python3
"""Run a provenance-checked compatibility smoke for the released scDFM model.

This is deliberately not a biological or leaderboard evaluation.  It loads the
unmodified upstream architecture and released weights, then performs one tiny
deterministic forward pass with synthetic log-expression values and a dummy
all-False interaction graph.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import linecache
import os
import platform
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from types import MappingProxyType, ModuleType
from typing import Any

import torch

from audit_scdfm_checkpoint import audit_checkpoint
from authenticate_scdfm_source import authenticate_source


SCHEMA = "vcc-scdfm-released-checkpoint-smoke-v2"
EXPECTED_VOCAB_SIZE = 5033
GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}")
ARCHITECTURE = {
    "ntoken": 6000,
    "d_model": 512,
    "nhead": 8,
    "d_hid": 2048,
    "nlayers": 4,
    "dropout": 0.1,
    "fusion_method": "differential_perceiver",
    "perturbation_function": "crisper",
    "use_perturbation_interaction": True,
}


class ReleasedCheckpointSmokeError(RuntimeError):
    """Raised when the released-checkpoint compatibility contract fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReleasedCheckpointSmokeError(message)


@dataclass(frozen=True)
class GitBlobSnapshot:
    """One immutable file read from a content-addressed Git blob."""

    path: str
    mode: str
    object_sha1: str
    sha256: str
    data: bytes


@dataclass(frozen=True)
class GitModuleSnapshot:
    """One importable Python module backed by a verified Git blob."""

    name: str
    blob: GitBlobSnapshot
    is_package: bool


def _git_bytes(source: Path, *arguments: str) -> bytes:
    """Run one read-only Git object query and return its exact bytes."""

    try:
        completed = subprocess.run(
            ["git", "-C", str(source), *arguments],
            check=False,
            capture_output=True,
        )
    except OSError as error:
        raise ReleasedCheckpointSmokeError(f"Unable to execute Git: {error}") from error
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ReleasedCheckpointSmokeError(
            f"Git object query failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout


def _git_blob_sha1(data: bytes) -> str:
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


class VerifiedGitModuleSnapshot(
    importlib.abc.MetaPathFinder,
    importlib.abc.InspectLoader,
):
    """Load upstream modules only from verified, immutable Git blob bytes."""

    def __init__(
        self,
        *,
        commit: str,
        tree: str,
        blobs: Mapping[str, GitBlobSnapshot],
    ) -> None:
        self.commit = commit
        self.tree = tree
        self._blobs = MappingProxyType(dict(blobs))
        modules: dict[str, GitModuleSnapshot] = {}
        package_names: set[str] = set()
        parent_packages: set[str] = set()
        for path, blob in self._blobs.items():
            if not path.endswith(".py"):
                continue
            parts = list(PurePosixPath(path).with_suffix("").parts)
            is_package = parts[-1] == "__init__"
            if is_package:
                parts.pop()
            name = ".".join(parts)
            _require(name and name not in modules, f"Duplicate snapshot module: {name}")
            modules[name] = GitModuleSnapshot(name, blob, is_package)
            if is_package:
                package_names.add(name)
            for index in range(1, len(parts)):
                parent_packages.add(".".join(parts[:index]))
        self._modules = MappingProxyType(modules)
        self._namespace_packages = frozenset(parent_packages - package_names)
        self._executed: dict[str, GitModuleSnapshot] = {}
        self._loaded_modules: dict[str, ModuleType] = {}
        self._loaded_module_audit_passed = False
        self._audited_loaded_modules: tuple[str, ...] = ()

    @classmethod
    def from_repository(
        cls,
        source: Path,
        *,
        expected_commit: str,
        expected_tree: str,
    ) -> "VerifiedGitModuleSnapshot":
        """Capture and verify every file below ``src`` from a pinned Git tree."""

        source = source.resolve()
        _require(
            GIT_OBJECT_PATTERN.fullmatch(expected_commit) is not None,
            "Pinned source commit must be a 40-character Git object ID",
        )
        _require(
            GIT_OBJECT_PATTERN.fullmatch(expected_tree) is not None,
            "Pinned source tree must be a 40-character Git object ID",
        )
        observed_commit = _git_bytes(
            source, "rev-parse", f"{expected_commit}^{{commit}}"
        ).decode("ascii").strip()
        observed_tree = _git_bytes(
            source, "rev-parse", f"{expected_commit}^{{tree}}"
        ).decode("ascii").strip()
        _require(observed_commit == expected_commit, "Pinned source commit object changed")
        _require(observed_tree == expected_tree, "Pinned source tree object changed")

        listing = _git_bytes(
            source,
            "ls-tree",
            "-r",
            "-z",
            "--full-tree",
            expected_commit,
            "--",
            "src",
        )
        blobs: dict[str, GitBlobSnapshot] = {}
        for raw_record in listing.split(b"\0"):
            if not raw_record:
                continue
            try:
                raw_metadata, raw_path = raw_record.split(b"\t", 1)
                mode, object_type, object_sha1 = raw_metadata.decode("ascii").split()
                path = raw_path.decode("utf-8")
            except (UnicodeError, ValueError) as error:
                raise ReleasedCheckpointSmokeError(
                    "Pinned Git tree contains an invalid ls-tree record"
                ) from error
            relative = PurePosixPath(path)
            _require(
                not relative.is_absolute()
                and ".." not in relative.parts
                and relative.parts
                and relative.parts[0] == "src",
                f"Unsafe path in pinned Git tree: {path!r}",
            )
            _require(path not in blobs, f"Duplicate path in pinned Git tree: {path!r}")
            _require(object_type == "blob", f"Non-blob entry below src: {path!r}")
            _require(mode in {"100644", "100755"}, f"Unsafe Git mode for {path!r}: {mode}")
            _require(
                GIT_OBJECT_PATTERN.fullmatch(object_sha1) is not None,
                f"Invalid blob object ID for {path!r}",
            )
            data = _git_bytes(source, "cat-file", "blob", object_sha1)
            _require(
                _git_blob_sha1(data) == object_sha1,
                f"Git blob content hash mismatch for {path!r}",
            )
            blobs[path] = GitBlobSnapshot(
                path=path,
                mode=mode,
                object_sha1=object_sha1,
                sha256=hashlib.sha256(data).hexdigest(),
                data=data,
            )
        _require(blobs, "Pinned Git tree contains no src files")
        _require(
            "src/models/origin/model.py" in blobs,
            "Pinned Git tree is missing the released model module",
        )
        return cls(commit=expected_commit, tree=expected_tree, blobs=blobs)

    def read_blob(self, path: str) -> GitBlobSnapshot:
        try:
            return self._blobs[path]
        except KeyError as error:
            raise ReleasedCheckpointSmokeError(
                f"Path is absent from the pinned Git snapshot: {path}"
            ) from error

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        record = self._modules.get(fullname)
        if record is not None:
            origin = f"git-object://{self.commit}/{record.blob.path}"
            spec = importlib.machinery.ModuleSpec(
                fullname,
                self,
                origin=origin,
                is_package=record.is_package,
            )
            if record.is_package:
                spec.submodule_search_locations = []
            return spec
        if fullname in self._namespace_packages:
            spec = importlib.machinery.ModuleSpec(
                fullname,
                loader=None,
                origin=f"git-tree://{self.tree}/{fullname.replace('.', '/')}",
                is_package=True,
            )
            spec.submodule_search_locations = []
            return spec
        return None

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType | None:
        del spec
        return None

    def is_package(self, fullname: str) -> bool:
        record = self._modules.get(fullname)
        if record is None:
            raise ImportError(f"Snapshot has no source for module: {fullname}")
        return record.is_package

    def get_filename(self, fullname: str) -> str:
        record = self._modules.get(fullname)
        if record is None:
            raise ImportError(f"Snapshot has no source for module: {fullname}")
        return f"git-object://{self.commit}/{record.blob.path}"

    def get_source(self, fullname: str) -> str:
        record = self._modules.get(fullname)
        if record is None:
            raise ImportError(f"Snapshot has no source for module: {fullname}")
        return importlib.util.decode_source(record.blob.data)

    def get_code(self, fullname: str):
        return compile(
            self.get_source(fullname),
            self.get_filename(fullname),
            "exec",
            dont_inherit=True,
        )

    def exec_module(self, module: ModuleType) -> None:
        name = module.__spec__.name
        record = self._modules.get(name)
        _require(record is not None, f"Snapshot has no source for module: {name}")
        origin = f"git-object://{self.commit}/{record.blob.path}"
        module.__file__ = origin
        module.__cached__ = None
        self._executed[name] = record
        self._loaded_modules[name] = module
        source = self.get_source(name)
        linecache.cache[origin] = (
            len(record.blob.data),
            None,
            source.splitlines(keepends=True),
            origin,
        )
        code = compile(source, origin, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    @contextlib.contextmanager
    def installed(self):
        """Install the snapshot finder and reject every preloaded src module."""

        preloaded = sorted(
            name for name in sys.modules if name == "src" or name.startswith("src.")
        )
        _require(not preloaded, f"Refusing preloaded upstream modules: {preloaded}")
        self._executed.clear()
        self._loaded_modules.clear()
        self._loaded_module_audit_passed = False
        self._audited_loaded_modules = ()
        sys.meta_path.insert(0, self)
        try:
            yield self
            self.audit_loaded_modules()
        finally:
            try:
                sys.meta_path.remove(self)
            except ValueError:
                pass
            for name in list(sys.modules):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            for record in self._executed.values():
                linecache.cache.pop(
                    f"git-object://{self.commit}/{record.blob.path}",
                    None,
                )

    def audit_loaded_modules(self) -> tuple[str, ...]:
        """Fail closed unless every loaded ``src`` module is snapshot-bound."""

        loaded = {
            name: value
            for name, value in sys.modules.items()
            if name == "src" or name.startswith("src.")
        }
        _require(loaded, "No upstream modules were loaded from the Git snapshot")
        for name, loaded_module in sorted(loaded.items()):
            spec = getattr(loaded_module, "__spec__", None)
            _require(spec is not None, f"Loaded upstream module has no spec: {name}")
            if name in self._modules:
                record = self._executed.get(name)
                expected_module = self._loaded_modules.get(name)
                _require(
                    record is not None
                    and expected_module is loaded_module
                    and getattr(loaded_module, "__loader__", None) is self
                    and getattr(spec, "loader", None) is self,
                    f"Loaded upstream module is unverified or replaced: {name}",
                )
                expected_origin = (
                    f"git-object://{self.commit}/{record.blob.path}"
                )
                _require(
                    getattr(spec, "origin", None) == expected_origin
                    and getattr(loaded_module, "__file__", None) == expected_origin,
                    f"Loaded upstream module origin changed: {name}",
                )
            elif name in self._namespace_packages:
                expected_origin = (
                    f"git-tree://{self.tree}/{name.replace('.', '/')}"
                )
                _require(
                    getattr(spec, "origin", None) == expected_origin
                    and getattr(loaded_module, "__file__", None) is None
                    and not list(getattr(loaded_module, "__path__", ())),
                    f"Loaded upstream namespace package is unverified: {name}",
                )
            else:
                raise ReleasedCheckpointSmokeError(
                    f"Loaded upstream module is absent from the pinned snapshot: {name}"
                )

        for name, expected_module in self._loaded_modules.items():
            _require(
                loaded.get(name) is expected_module,
                f"Executed upstream module was removed or replaced: {name}",
            )
        self._audited_loaded_modules = tuple(sorted(loaded))
        self._loaded_module_audit_passed = True
        return self._audited_loaded_modules

    def require_loaded_module(self, module: ModuleType) -> str:
        name = getattr(module, "__name__", None)
        _require(
            isinstance(name, str)
            and name in self._executed
            and getattr(module, "__loader__", None) is self,
            "Upstream module was not executed by the verified Git-object loader",
        )
        return f"git-object://{self.commit}/{self._executed[name].blob.path}"

    def execution_receipt(self) -> dict[str, Any]:
        _require(
            self._loaded_module_audit_passed,
            "Loaded upstream modules were not audited before receipt creation",
        )
        return {
            "commit": self.commit,
            "tree": self.tree,
            "snapshot_scope": "complete_src_tree",
            "src_blob_count": len(self._blobs),
            "python_module_count": len(self._modules),
            "all_git_blob_sha1_values_verified": True,
            "loaded_module_audit_passed": True,
            "all_loaded_src_modules_byte_bound": True,
            "audited_loaded_modules": list(self._audited_loaded_modules),
            "mutable_worktree_imported": False,
            "loader": "verified_in_memory_git_blob_loader",
            "executed_modules": [
                {
                    "module": name,
                    "path": record.blob.path,
                    "git_blob_sha1": record.blob.object_sha1,
                    "sha256": record.blob.sha256,
                }
                for name, record in sorted(self._executed.items())
            ],
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def authenticate_norman_vocab(
    vocab_path: Path,
    *,
    source_root: Path,
    expected_size: int = EXPECTED_VOCAB_SIZE,
) -> dict[str, Any]:
    """Validate the pinned Norman vocabulary without importing its tokenizer."""

    source_root = source_root.resolve()
    unexpanded_vocab_path = vocab_path.expanduser().absolute()
    _require(
        not unexpanded_vocab_path.is_symlink(),
        "Norman vocabulary must not be a symlink",
    )
    vocab_path = unexpanded_vocab_path.resolve()
    _require(vocab_path.is_file(), f"Missing Norman vocabulary: {vocab_path}")
    try:
        vocab_path.relative_to(source_root)
    except ValueError as error:
        raise ReleasedCheckpointSmokeError(
            "Norman vocabulary must be inside the authenticated source tree"
        ) from error

    try:
        encoded = vocab_path.read_bytes()
    except OSError as error:
        raise ReleasedCheckpointSmokeError(f"Invalid Norman vocabulary: {error}") from error
    return _validate_norman_vocab_bytes(
        encoded,
        display_path=str(vocab_path),
        expected_size=expected_size,
        provenance="authenticated_worktree_file",
    )


def _validate_norman_vocab_bytes(
    encoded: bytes,
    *,
    display_path: str,
    expected_size: int,
    provenance: str,
) -> dict[str, Any]:
    """Validate exact vocabulary bytes and return a byte-bound descriptor."""

    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ReleasedCheckpointSmokeError(f"Invalid Norman vocabulary: {error}") from error
    _require(isinstance(payload, dict), "Norman vocabulary root must be an object")
    _require(
        all(isinstance(token, str) and isinstance(index, int) for token, index in payload.items()),
        "Norman vocabulary must map strings to integer token IDs",
    )
    _require(len(payload) == expected_size, f"Expected {expected_size} Norman tokens")
    _require(
        sorted(payload.values()) == list(range(expected_size)),
        "Norman vocabulary IDs must be unique and contiguous from zero",
    )
    expected_specials = {"<pad>": 0, "<cls>": 1, "<mask>": 2, "control": 3}
    _require(
        all(payload.get(token) == index for token, index in expected_specials.items()),
        "Norman vocabulary special-token IDs do not match the released convention",
    )
    return {
        "path": display_path,
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "provenance": provenance,
        "token_count": len(payload),
        "minimum_token_id": min(payload.values()),
        "maximum_token_id": max(payload.values()),
        "special_tokens": expected_specials,
    }


def authenticate_norman_vocab_snapshot(
    vocab_path: Path,
    *,
    source_root: Path,
    snapshot: VerifiedGitModuleSnapshot,
    expected_size: int = EXPECTED_VOCAB_SIZE,
) -> dict[str, Any]:
    """Authenticate vocabulary bytes from the same pinned Git snapshot as code."""

    source_root = source_root.resolve()
    lexical_path = Path(os.path.abspath(os.fspath(vocab_path.expanduser())))
    _require(not lexical_path.is_symlink(), "Norman vocabulary must not be a symlink")
    resolved_path = lexical_path.resolve()
    try:
        relative_path = resolved_path.relative_to(source_root).as_posix()
    except ValueError as error:
        raise ReleasedCheckpointSmokeError(
            "Norman vocabulary must be inside the authenticated source tree"
        ) from error
    blob = snapshot.read_blob(relative_path)
    receipt = _validate_norman_vocab_bytes(
        blob.data,
        display_path=f"git-object://{snapshot.commit}/{relative_path}",
        expected_size=expected_size,
        provenance="verified_git_blob_snapshot",
    )
    receipt["git_blob_sha1"] = blob.object_sha1
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate safety claims before publishing the smoke receipt."""

    _require(receipt.get("schema") == SCHEMA, "Unexpected receipt schema")
    _require(receipt.get("status") == "passed", "Smoke receipt did not pass")
    checks = receipt.get("checks")
    _require(isinstance(checks, Mapping) and checks, "Receipt checks are missing")
    _require(all(value is True for value in checks.values()), "Not every smoke check passed")
    scope = receipt.get("scope")
    _require(isinstance(scope, Mapping), "Receipt scope is missing")
    _require(scope.get("synthetic_expression_only") is True, "Synthetic scope is not explicit")
    _require(scope.get("dummy_all_false_graph") is True, "Dummy graph scope is not explicit")
    _require(scope.get("model_performance_evaluation") is False, "Performance claim is forbidden")
    _require(scope.get("official_submission_artifact") is False, "Submission claim is forbidden")


def write_json_atomic_no_overwrite(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish JSON atomically while refusing every overwrite race."""

    validate_receipt(payload)
    path = Path(os.path.abspath(os.fspath(path.expanduser())))
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to overwrite receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise FileExistsError(f"Refusing to overwrite receipt: {path}") from error
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _device_receipt(device: torch.device) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(device)
    return {
        "type": device.type,
        "index": device.index if device.index is not None else torch.cuda.current_device(),
        "name": properties.name,
        "total_memory_bytes": properties.total_memory,
        "compute_capability": f"{properties.major}.{properties.minor}",
    }


def _load_authenticated_model_state_dict(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
) -> tuple[Mapping[str, torch.Tensor], str]:
    """Hash and restricted-load one immutable descriptor, then return model state."""

    path = Path(os.path.abspath(os.fspath(checkpoint_path.expanduser())))
    try:
        initial = path.lstat()
    except OSError as error:
        raise ReleasedCheckpointSmokeError(
            f"Unable to inspect checkpoint for model loading: {error}"
        ) from error
    _require(stat.S_ISREG(initial.st_mode), "Checkpoint must be a regular file")
    _require(not stat.S_ISLNK(initial.st_mode), "Checkpoint must not be a symlink")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ReleasedCheckpointSmokeError(
            f"Unable to open checkpoint for model loading: {error}"
        ) from error
    with os.fdopen(descriptor, "rb") as handle:
        opened = os.fstat(handle.fileno())
        _require(
            stat.S_ISREG(opened.st_mode)
            and (opened.st_dev, opened.st_ino) == (initial.st_dev, initial.st_ino),
            "Checkpoint changed while it was opened for model loading",
        )
        digest = hashlib.sha256()
        while block := handle.read(8 << 20):
            digest.update(block)
        observed_sha256 = digest.hexdigest()
        _require(
            observed_sha256 == expected_sha256,
            "Checkpoint SHA-256 changed before model loading",
        )
        handle.seek(0)
        try:
            checkpoint = torch.load(handle, map_location="cpu", weights_only=True)
        except Exception as error:
            raise ReleasedCheckpointSmokeError(
                f"Restricted model-state loading failed: {type(error).__name__}: {error}"
            ) from error
        after = os.fstat(handle.fileno())
        _require(
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            == (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns),
            "Checkpoint changed during model-state loading",
        )

    _require(isinstance(checkpoint, Mapping), "Checkpoint root must be a mapping")
    state = checkpoint.get("model_state_dict")
    _require(isinstance(state, Mapping), "Checkpoint model_state_dict is missing")
    _require(all(isinstance(key, str) for key in state), "Model-state keys must be strings")
    _require(all(isinstance(value, torch.Tensor) for value in state.values()), "Model state must contain tensors only")
    return state, observed_sha256


def run_smoke(
    *,
    source: Path,
    checkpoint: Path,
    vocab: Path,
    expected_checkpoint_sha256: str,
    batch_size: int = 2,
    gene_count: int = 8,
    seed: int = 20260901,
) -> dict[str, Any]:
    """Authenticate assets, strict-load the model, and run one H100 forward."""

    _require(torch.cuda.is_available(), "A CUDA device allocated by Slurm is required")
    device = torch.device("cuda", 0)
    device_info = _device_receipt(device)
    _require("H100" in device_info["name"].upper(), f"An H100 is required; found {device_info['name']}")
    _require(batch_size > 0 and gene_count > 1, "Positive batch size and at least two genes are required")

    source_receipt = authenticate_source(source)
    source_details = source_receipt.get("source")
    _require(isinstance(source_details, Mapping), "Source receipt details are missing")
    source_snapshot = VerifiedGitModuleSnapshot.from_repository(
        source,
        expected_commit=str(source_details.get("repository_commit")),
        expected_tree=str(source_details.get("repository_tree")),
    )
    checkpoint_receipt = audit_checkpoint(
        checkpoint,
        expected_sha256=expected_checkpoint_sha256,
    )
    vocab_receipt = authenticate_norman_vocab_snapshot(
        vocab,
        source_root=source,
        snapshot=source_snapshot,
    )
    vocab_size = int(vocab_receipt["token_count"])

    with source_snapshot.installed():
        upstream_module = importlib.import_module("src.models.origin.model")
        upstream_module_origin = source_snapshot.require_loaded_module(upstream_module)
        upstream_model_class = getattr(upstream_module, "model")

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True)
        with tempfile.TemporaryDirectory(prefix="vcc-scdfm-dummy-mask-") as temporary:
            mask_path = Path(temporary) / "dummy_graph" / "all_false_mask.pt"
            mask_path.parent.mkdir(parents=True)
            dummy_mask = torch.zeros((vocab_size, vocab_size), dtype=torch.bool)
            torch.save(dummy_mask, mask_path)
            del dummy_mask

            model = upstream_model_class(**ARCHITECTURE, mask_path=str(mask_path))
            state, loaded_checkpoint_sha256 = _load_authenticated_model_state_dict(
                checkpoint,
                expected_sha256=expected_checkpoint_sha256,
            )
            incompatible = model.load_state_dict(state, strict=True)
            del state
            model = model.to(device).eval()

            generator = torch.Generator(device="cpu").manual_seed(seed)
            gene_ids_cpu = torch.arange(4, 4 + gene_count, dtype=torch.long).repeat(batch_size, 1)
            target_ids_cpu = torch.arange(4 + gene_count, 4 + gene_count + batch_size, dtype=torch.long).unsqueeze(1)
            control_cpu = torch.rand((batch_size, gene_count), generator=generator) * 4.0
            intermediate_cpu = control_cpu + 0.05 * torch.randn(
                (batch_size, gene_count), generator=generator
            )
            time_cpu = torch.linspace(0.2, 0.8, batch_size)
            inputs = {
                "gene_id": gene_ids_cpu.to(device),
                "cell_1": intermediate_cpu.to(device),
                "t": time_cpu.to(device),
                "cell_2": control_cpu.to(device),
                "perturbation_id": target_ids_cpu.to(device),
            }
            with torch.inference_mode():
                first = model(**inputs)
                second = model(**inputs)
            torch.cuda.synchronize(device)
    source_snapshot_receipt = source_snapshot.execution_receipt()

    expected_shape = (batch_size, gene_count)
    finite = bool(torch.isfinite(first).all().item())
    repeat_equal = bool(torch.equal(first, second))
    shape_matches = tuple(first.shape) == expected_shape
    output_cpu = first.detach().float().cpu().contiguous()
    output_sha256 = hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest()
    strict_clean = not incompatible.missing_keys and not incompatible.unexpected_keys

    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "passed",
        "scope": {
            "synthetic_expression_only": True,
            "dummy_all_false_graph": True,
            "dummy_graph_has_biological_information": False,
            "challenge_expression_data_read": False,
            "model_performance_evaluation": False,
            "official_submission_artifact": False,
        },
        "checks": {
            "source_authentication_passed": source_receipt["status"] == "passed",
            "checkpoint_audit_passed": checkpoint_receipt["status"] == "passed",
            "checkpoint_loaded_with_weights_only": True,
            "upstream_modules_loaded_only_from_verified_git_blobs": (
                source_snapshot_receipt["mutable_worktree_imported"] is False
                and source_snapshot_receipt["loaded_module_audit_passed"] is True
                and source_snapshot_receipt["all_loaded_src_modules_byte_bound"] is True
                and bool(source_snapshot_receipt["executed_modules"])
            ),
            "loaded_checkpoint_sha256_matches_audit": (
                loaded_checkpoint_sha256
                == checkpoint_receipt["checkpoint"]["sha256"]
            ),
            "norman_vocab_has_expected_5033_rows": vocab_size == EXPECTED_VOCAB_SIZE,
            "strict_state_dict_load_has_no_missing_or_unexpected_keys": strict_clean,
            "output_shape_matches": shape_matches,
            "output_values_are_finite": finite,
            "repeated_eval_forward_is_bitwise_identical": repeat_equal,
            "h100_device_used": "H100" in device_info["name"].upper(),
        },
        "provenance": {
            "source": source_receipt,
            "source_execution_snapshot": source_snapshot_receipt,
            "checkpoint": checkpoint_receipt,
            "vocabulary": vocab_receipt,
        },
        "model": {
            "upstream_import": "src.models.origin.model.model",
            "upstream_module_origin": upstream_module_origin,
            "mutable_worktree_imported": False,
            "upstream_source_modified": False,
            "architecture": ARCHITECTURE,
            "strict_load": True,
            "loaded_checkpoint_sha256": loaded_checkpoint_sha256,
            "missing_keys": list(incompatible.missing_keys),
            "unexpected_keys": list(incompatible.unexpected_keys),
            "optimizer_state_restored": False,
            "scheduler_state_restored": False,
            "split_pickle_read": False,
        },
        "synthetic_forward": {
            "seed": seed,
            "batch_size": batch_size,
            "gene_count": gene_count,
            "gene_id_range": [4, 3 + gene_count],
            "target_id_range": [4 + gene_count, 3 + gene_count + batch_size],
            "input_semantics": "synthetic nonnegative log-expression-like values",
            "output_shape": list(first.shape),
            "output_dtype": str(first.dtype).removeprefix("torch."),
            "output_sha256_float32_cpu_bytes": output_sha256,
            "output_min": float(output_cpu.min().item()),
            "output_max": float(output_cpu.max().item()),
        },
        "device": device_info,
        "software": {
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "scheduler": {
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_job_name": os.environ.get("SLURM_JOB_NAME"),
        },
    }
    validate_receipt(receipt)
    return receipt


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gene-count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260901)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    receipt = run_smoke(
        source=args.source,
        checkpoint=args.checkpoint,
        vocab=args.vocab,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        batch_size=args.batch_size,
        gene_count=args.gene_count,
        seed=args.seed,
    )
    write_json_atomic_no_overwrite(args.output_json, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


if __name__ == "__main__":
    main()
