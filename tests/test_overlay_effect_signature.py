"""Focused tests for effect-signature overlay alignment and provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from calibrate_cross_context_effects import load_effect_atlas  # noqa: E402
from overlay_effect_signature import run  # noqa: E402


def arguments(directory: Path) -> argparse.Namespace:
    return argparse.Namespace(
        bayesian_fallback=directory / "fallback.npz",
        signature_prior=directory / "signature.npz",
        challenge_targets=directory / "targets.csv",
        challenge_genes=directory / "genes.csv",
        output_npz=directory / "overlay.npz",
        output_json=directory / "overlay.json",
        target_column="target_gene",
        gene_column="gene_name",
    )


def write_axes(directory: Path) -> None:
    pd.DataFrame({"target_gene": ["t0", "t1", "t2"]}).to_csv(
        directory / "targets.csv", index=False
    )
    pd.DataFrame({"gene_name": ["g0", "g1", "g2", "g3"]}).to_csv(
        directory / "genes.csv", index=False
    )


class OverlayIntegrationTests(unittest.TestCase):
    def test_name_alignment_masks_and_fallback_preservation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_axes(directory)
            fallback_targets = ["extra_t", "t2", "t0", "t1"]
            fallback_genes = ["g2", "g0", "extra_g", "g3", "g1"]
            fallback = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
            np.savez_compressed(
                directory / "fallback.npz",
                effects=fallback,
                contexts=np.asarray(["A", "B"]),
                targets=np.asarray(fallback_targets),
                genes=np.asarray(fallback_genes),
            )
            signature = np.asarray(
                [
                    [[900.0, 901.0], [920.0, 921.0]],
                    [[800.0, 801.0], [820.0, 821.0]],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                directory / "signature.npz",
                effects=signature,
                contexts=np.asarray(["B", "A"]),
                targets=np.asarray(["t1", "t0"]),
                genes=np.asarray(["g3", "g1"]),
                effect_target_mask=np.asarray([True, False]),
                effect_gene_mask=np.asarray([True, False]),
                effect_mask=np.asarray(
                    [[True, False], [False, False]]
                ),
            )
            signature_hash = hashlib.sha256(
                (directory / "signature.npz").read_bytes()
            ).hexdigest()
            (directory / "signature.json").write_text(
                json.dumps(
                    {
                        "schema": "synthetic-signature-v1",
                        "outputs": {"effect_prior": {"sha256": signature_hash}},
                    }
                ),
                encoding="utf-8",
            )
            args = arguments(directory)
            report = run(args)

            fallback_target_lookup = {
                name: index for index, name in enumerate(fallback_targets)
            }
            fallback_gene_lookup = {name: index for index, name in enumerate(fallback_genes)}
            expected = fallback[
                np.ix_(
                    np.arange(2),
                    [fallback_target_lookup[name] for name in ["t0", "t1", "t2"]],
                    [fallback_gene_lookup[name] for name in ["g0", "g1", "g2", "g3"]],
                )
            ].copy()
            expected[0, 1, 3] = 800.0
            expected[1, 1, 3] = 900.0
            with np.load(args.output_npz, allow_pickle=False) as artifact:
                np.testing.assert_array_equal(artifact["effects"], expected)
                np.testing.assert_array_equal(artifact["targets"], ["t0", "t1", "t2"])
                np.testing.assert_array_equal(artifact["genes"], ["g0", "g1", "g2", "g3"])
                self.assertEqual(int(artifact["signature_overlay_mask"].sum()), 2)
                np.testing.assert_array_equal(
                    artifact["signature_overlay_target_mask"], [False, True, False]
                )
                np.testing.assert_array_equal(
                    artifact["signature_overlay_gene_mask"], [False, False, False, True]
                )
            loaded = load_effect_atlas(args.output_npz, require_cell_counts=True)
            self.assertEqual(loaded.effects.shape, (3, 4))
            np.testing.assert_allclose(loaded.effects, expected.mean(axis=0))
            self.assertEqual(report["overlay"]["written_coordinates"], 2)
            self.assertEqual(report["overlay"]["overlaid_target_gene_coordinates"], 1)
            self.assertEqual(report["overlay"]["preserved_fallback_coordinates"], 22)
            self.assertEqual(
                report["output"]["sha256"],
                hashlib.sha256(args.output_npz.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                report["inputs"]["signature_prior_metadata"]["schema"],
                "synthetic-signature-v1",
            )
            parsed = json.loads(args.output_json.read_text(encoding="utf-8"))
            self.assertTrue(parsed["contract"]["fallback_preserved_outside_overlay"])

    def test_context_independent_signature_broadcasts_to_fallback_contexts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_axes(directory)
            fallback = np.ones((2, 3, 4), dtype=np.float32)
            np.savez_compressed(
                directory / "fallback.npz",
                effects=fallback,
                contexts=np.asarray(["A", "B"]),
                targets=np.asarray(["t0", "t1", "t2"]),
                genes=np.asarray(["g0", "g1", "g2", "g3"]),
            )
            np.savez_compressed(
                directory / "signature.npz",
                effects=np.asarray([[0.0, 7.0]], dtype=np.float32),
                targets=np.asarray([b"t2"]),
                target_names=np.asarray([b"t2"]),
                genes=np.asarray([b"g0", b"g1"]),
                gene_names=np.asarray([b"g0", b"g1"]),
            )
            args = arguments(directory)
            report = run(args)
            with np.load(args.output_npz, allow_pickle=False) as artifact:
                np.testing.assert_array_equal(artifact["effects"][:, 2, 0], [0.0, 0.0])
                np.testing.assert_array_equal(artifact["effects"][:, 2, 1], [7.0, 7.0])
                self.assertEqual(int(artifact["signature_overlay_mask"].sum()), 4)
            self.assertEqual(report["overlay"]["written_coordinates_by_context"], {"A": 2, "B": 2})
            self.assertEqual(report["overlay"]["overlaid_target_gene_coordinates"], 2)

    def test_signature_axis_outside_challenge_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_axes(directory)
            np.savez_compressed(
                directory / "fallback.npz",
                effects=np.zeros((3, 4), dtype=np.float32),
                targets=np.asarray(["t0", "t1", "t2"]),
                genes=np.asarray(["g0", "g1", "g2", "g3"]),
            )
            np.savez_compressed(
                directory / "signature.npz",
                effects=np.ones((1, 1), dtype=np.float32),
                targets=np.asarray(["outside"]),
                genes=np.asarray(["g0"]),
            )
            args = arguments(directory)
            with self.assertRaisesRegex(ValueError, "outside challenge axis"):
                run(args)

    def test_missing_fallback_challenge_axis_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_axes(directory)
            np.savez_compressed(
                directory / "fallback.npz",
                effects=np.zeros((3, 3), dtype=np.float32),
                targets=np.asarray(["t0", "t1", "t2"]),
                genes=np.asarray(["g0", "g1", "g2"]),
            )
            np.savez_compressed(
                directory / "signature.npz",
                effects=np.ones((1, 1), dtype=np.float32),
                targets=np.asarray(["t0"]),
                genes=np.asarray(["g0"]),
            )
            args = arguments(directory)
            with self.assertRaisesRegex(ValueError, "Fallback misses challenge genes"):
                run(args)
            self.assertFalse(args.output_npz.exists())
            self.assertFalse(args.output_json.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            write_axes(directory)
            np.savez_compressed(
                directory / "fallback.npz",
                effects=np.zeros((3, 4), dtype=np.float32),
                targets=np.asarray(["t0", "t1", "t2"]),
                genes=np.asarray(["g0", "g1", "g2", "g3"]),
            )
            np.savez_compressed(
                directory / "signature.npz",
                effects=np.ones((1, 1), dtype=np.float32),
                targets=np.asarray(["t0"]),
                genes=np.asarray(["g0"]),
            )
            args = arguments(directory)
            args.output_npz.write_bytes(b"keep-me")
            with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
                run(args)
            self.assertEqual(args.output_npz.read_bytes(), b"keep-me")
            self.assertFalse(args.output_json.exists())


if __name__ == "__main__":
    unittest.main()
