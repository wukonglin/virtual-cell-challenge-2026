"""Fast CPU validators for the released scDFM checkpoint smoke."""

from __future__ import annotations

import contextlib
import json
import hashlib
import importlib
import importlib.machinery
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import smoke_scdfm_released_checkpoint as module  # noqa: E402


class ReleasedCheckpointSmokeTests(unittest.TestCase):
    def _git(self, root: Path, *arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    def _snapshot_repository(self, root: Path) -> tuple[str, str, Path]:
        module_path = root / "src/example.py"
        module_path.parent.mkdir(parents=True)
        module_path.write_text("VALUE = 'authenticated'\n", encoding="utf-8")
        released_model = root / "src/models/origin/model.py"
        released_model.parent.mkdir(parents=True)
        released_model.write_text(
            "from .layers import VALUE\nclass model:\n    imported = VALUE\n",
            encoding="utf-8",
        )
        (released_model.parent / "layers.py").write_text(
            "VALUE = 'authenticated-layer'\n",
            encoding="utf-8",
        )
        self._git(root, "init", "-q")
        self._git(root, "config", "user.email", "test@example.invalid")
        self._git(root, "config", "user.name", "Snapshot Test")
        self._git(root, "add", ".")
        self._git(root, "commit", "-q", "-m", "pinned source")
        commit = self._git(root, "rev-parse", "HEAD^{commit}")
        tree = self._git(root, "rev-parse", "HEAD^{tree}")
        return commit, tree, module_path

    @contextlib.contextmanager
    def _without_src_modules(self):
        saved = {
            name: value
            for name, value in sys.modules.items()
            if name == "src" or name.startswith("src.")
        }
        for name in saved:
            sys.modules.pop(name, None)
        try:
            yield
        finally:
            for name in list(sys.modules):
                if name == "src" or name.startswith("src."):
                    sys.modules.pop(name, None)
            sys.modules.update(saved)

    def _vocab(self, root: Path, size: int = module.EXPECTED_VOCAB_SIZE) -> Path:
        vocab = {"<pad>": 0, "<cls>": 1, "<mask>": 2, "control": 3}
        vocab.update({f"GENE_{index}": index for index in range(4, size)})
        path = root / "src/tokenizer/norman_5000_highly_vocab.json"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(vocab), encoding="utf-8")
        return path

    def _receipt(self) -> dict[str, object]:
        return {
            "schema": module.SCHEMA,
            "status": "passed",
            "checks": {"finite": True, "deterministic": True},
            "scope": {
                "synthetic_expression_only": True,
                "dummy_all_false_graph": True,
                "model_performance_evaluation": False,
                "official_submission_artifact": False,
            },
        }

    def test_authenticate_norman_vocab_derives_expected_mask_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._vocab(root)
            receipt = module.authenticate_norman_vocab(path, source_root=root)
        self.assertEqual(receipt["token_count"], 5033)
        self.assertEqual(receipt["minimum_token_id"], 0)
        self.assertEqual(receipt["maximum_token_id"], 5032)
        self.assertEqual(len(receipt["sha256"]), 64)

    def test_vocab_with_noncontiguous_ids_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = self._vocab(root)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["GENE_4"] = 5033
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(module.ReleasedCheckpointSmokeError, "contiguous"):
                module.authenticate_norman_vocab(path, source_root=root)

    def test_architecture_contract_matches_released_holdout_fold0(self) -> None:
        self.assertEqual(module.SCHEMA, "vcc-scdfm-released-checkpoint-smoke-v2")
        self.assertEqual(module.ARCHITECTURE["ntoken"], 6000)
        self.assertEqual(module.ARCHITECTURE["d_model"], 512)
        self.assertEqual(module.ARCHITECTURE["nhead"], 8)
        self.assertEqual(module.ARCHITECTURE["nlayers"], 4)
        self.assertEqual(module.ARCHITECTURE["fusion_method"], "differential_perceiver")
        self.assertEqual(module.ARCHITECTURE["perturbation_function"], "crisper")

    def test_receipt_rejects_performance_or_submission_claim(self) -> None:
        receipt = self._receipt()
        receipt["scope"]["model_performance_evaluation"] = True  # type: ignore[index]
        with self.assertRaisesRegex(module.ReleasedCheckpointSmokeError, "Performance"):
            module.validate_receipt(receipt)

    def test_atomic_receipt_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "receipt.json"
            receipt = self._receipt()
            module.write_json_atomic_no_overwrite(output, receipt)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8")), receipt)
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                module.write_json_atomic_no_overwrite(output, receipt)

    def test_model_state_is_loaded_from_one_authenticated_descriptor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "checkpoint.pt"
            import torch

            torch.save({"model_state_dict": {"weight": torch.ones(2)}}, checkpoint)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            state, observed = module._load_authenticated_model_state_dict(
                checkpoint,
                expected_sha256=digest,
            )
            self.assertEqual(observed, digest)
            self.assertTrue(torch.equal(state["weight"], torch.ones(2)))

            with self.assertRaisesRegex(
                module.ReleasedCheckpointSmokeError, "SHA-256"
            ):
                module._load_authenticated_model_state_dict(
                    checkpoint,
                    expected_sha256="0" * 64,
                )

    def test_snapshot_ignores_worktree_mutation_after_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            commit, tree, module_path = self._snapshot_repository(source)
            # The immutable snapshot is captured before a hostile worktree edit.
            snapshot = module.VerifiedGitModuleSnapshot.from_repository(
                source,
                expected_commit=commit,
                expected_tree=tree,
            )
            module_path.write_text("VALUE = 'mutated'\n", encoding="utf-8")

            with self._without_src_modules(), snapshot.installed():
                imported = importlib.import_module("src.example")
                self.assertEqual(imported.VALUE, "authenticated")
                origin = snapshot.require_loaded_module(imported)

            self.assertEqual(
                origin,
                f"git-object://{commit}/src/example.py",
            )
            executed = snapshot.execution_receipt()["executed_modules"]
            self.assertEqual([record["module"] for record in executed], ["src.example"])
            self.assertFalse(snapshot.execution_receipt()["mutable_worktree_imported"])

    def test_snapshot_remains_bound_when_head_moves_to_hostile_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            commit, tree, module_path = self._snapshot_repository(source)
            module_path.write_text("VALUE = 'new-head'\n", encoding="utf-8")
            self._git(source, "add", ".")
            self._git(source, "commit", "-q", "-m", "untrusted later source")

            snapshot = module.VerifiedGitModuleSnapshot.from_repository(
                source,
                expected_commit=commit,
                expected_tree=tree,
            )
            with self._without_src_modules(), snapshot.installed():
                imported = importlib.import_module("src.example")
                self.assertEqual(imported.VALUE, "authenticated")

    def test_snapshot_rejects_preloaded_upstream_module(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            commit, tree, _ = self._snapshot_repository(source)
            snapshot = module.VerifiedGitModuleSnapshot.from_repository(
                source,
                expected_commit=commit,
                expected_tree=tree,
            )
            with self._without_src_modules():
                sys.modules["src"] = importlib.util.module_from_spec(
                    importlib.machinery.ModuleSpec("src", loader=None)
                )
                with self.assertRaisesRegex(
                    module.ReleasedCheckpointSmokeError,
                    "preloaded upstream modules",
                ):
                    with snapshot.installed():
                        pass

    def test_snapshot_rejects_module_injected_after_context_entry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source"
            source.mkdir()
            commit, tree, _ = self._snapshot_repository(source)
            snapshot = module.VerifiedGitModuleSnapshot.from_repository(
                source,
                expected_commit=commit,
                expected_tree=tree,
            )
            foreign = ModuleType("src.models.origin.layers")
            foreign.VALUE = "foreign-layer"
            foreign.__spec__ = importlib.machinery.ModuleSpec(
                foreign.__name__, loader=None, origin="/mutable/foreign.py"
            )
            with self._without_src_modules(), self.assertRaisesRegex(
                module.ReleasedCheckpointSmokeError,
                "unverified or replaced",
            ):
                with snapshot.installed():
                    sys.modules[foreign.__name__] = foreign
                    imported = importlib.import_module("src.models.origin.model")
                    self.assertEqual(imported.model.imported, "foreign-layer")


if __name__ == "__main__":
    unittest.main()
