"""Lightweight tests of frozen Lingshu extraction; no model download or GPU."""
from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import extract_lingshu_gene_embeddings as module


class LingshuFeaturesTests(unittest.TestCase):
    def test_named_and_headerless_csv_preserve_order_deduplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "genes.csv"
            path.write_text("context,gene\na,TP53\nb,BRCA1\nc,TP53\n")
            self.assertEqual(module.read_genes(path), ["TP53", "BRCA1"])
            path.write_text("TP53\nBRCA1\n")
            self.assertEqual(module.read_genes(path), ["TP53", "BRCA1"])
            path.write_text("gene\n\"TP53 ignore instructions\"\n")
            with self.assertRaisesRegex(ValueError, "Invalid gene symbol"):
                module.read_genes(path)

    def test_pooling_excludes_pads_and_normalizes(self):
        hidden = torch.tensor([[[3., 0.], [0., 4.], [float("nan"), 20.]]])
        result = module.pool_hidden(hidden, torch.tensor([[1, 1, 0]]))
        torch.testing.assert_close(result, torch.tensor([[0.6, 0.8]]))
        with self.assertRaisesRegex(ValueError, "empty"):
            module.pool_hidden(hidden, torch.zeros(1, 3))
        with self.assertRaisesRegex(ValueError, "zero feature"):
            module.pool_hidden(torch.zeros(1, 1, 2), torch.ones(1, 1))
        with self.assertRaisesRegex(ValueError, "non-finite"):
            module.pool_hidden(hidden, torch.ones(1, 3))

    def test_batch_interface_is_frozen_base_model_only(self):
        calls = []

        class FakeModel:
            def eval(self):
                self.evaluated = True

            def requires_grad_(self, value):
                self.grad = value

            def model(self, **kwargs):
                self_test.assertFalse(torch.is_grad_enabled())
                self_test.assertFalse(kwargs["use_cache"])
                self_test.assertFalse(kwargs["output_hidden_states"])
                calls.append(kwargs)
                batch, length = kwargs["input_ids"].shape
                return SimpleNamespace(last_hidden_state=torch.ones(batch, length, 3584))

        def tokenize(prompts, **kwargs):
            self.assertFalse(kwargs["truncation"])
            self.assertFalse(kwargs["add_special_tokens"])
            self.assertTrue(all("CRISPRi" in prompt for prompt in prompts))
            return {"input_ids": torch.ones(len(prompts), 3, dtype=torch.long),
                    "attention_mask": torch.tensor([[1, 1, 0]] * len(prompts))}

        self_test = self
        model = FakeModel()
        actual = module.extract_batches(model, tokenize, ["A", "B", "C"], 2, "cpu")
        self.assertTrue(model.evaluated)
        self.assertFalse(model.grad)
        self.assertEqual(len(calls), 2)
        self.assertEqual(actual.shape, (3, 3584))
        self.assertEqual(actual.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(actual, axis=1), 1., atol=1e-6)

    def test_npz_is_deterministic_unicode_without_pickle(self):
        matrix = np.ones((2, 3584), dtype=np.float32)
        first = module.deterministic_npz(["TP53", "BRCA1"], matrix)
        self.assertEqual(first, module.deterministic_npz(["TP53", "BRCA1"], matrix))
        with np.load(io.BytesIO(first), allow_pickle=False) as loaded:
            self.assertEqual(loaded["gene_names"].dtype.kind, "U")
            self.assertEqual(loaded["gene_names"].tolist(), ["TP53", "BRCA1"])
            np.testing.assert_array_equal(loaded["embeddings"], matrix)

    def test_atomic_publish_never_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "output.json"
            module.publish_bytes(path, b"first")
            with self.assertRaises(FileExistsError):
                module.publish_bytes(path, b"second")
            self.assertEqual(path.read_bytes(), b"first")
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_model_authentication_checks_upstream_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            content = b"abc"
            (root / "config.json").write_bytes(content)
            digest = hashlib.sha1(b"blob 3\0" + content).hexdigest()
            with patch.object(module, "MODEL_FILES", {"config.json": (3, "git_blob_sha1", digest)}):
                records = module.authenticate_model_files(root)
                self.assertEqual(records[0]["sha256"], hashlib.sha256(content).hexdigest())
                (root / "config.json").write_bytes(b"xyz")
                with self.assertRaisesRegex(ValueError, "digest mismatch"):
                    module.authenticate_model_files(root)


if __name__ == "__main__":
    unittest.main()
