"""Regression tests for consumers of authenticated perturbation features."""

from __future__ import annotations

import hashlib
import pickle
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import yaml


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import authenticate_esm2_target_features as authenticator  # noqa: E402
import fit_bayesian_prior_h100 as bayesian  # noqa: E402
import generate_native_state_anchor_counts as native_counts  # noqa: E402
import generate_state_direct_counts as direct_counts  # noqa: E402
import generate_state_scorer_aware_counts as scorer_counts  # noqa: E402
import infer_state_effect_prior as state_inference  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_marker(path: str) -> dict[str, object]:
    Path(path).write_text("executed", encoding="utf-8")
    return {}


class _Exploit:
    def __init__(self, marker: Path) -> None:
        self.marker = marker

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return _write_marker, (str(self.marker),)


class _FakeStateModel:
    def __init__(self, **hyperparameters: object) -> None:
        self.hyperparameters = hyperparameters
        self.loaded: tuple[dict[str, torch.Tensor], bool] | None = None
        self.device: torch.device | None = None
        self.evaluating = False

    def load_state_dict(
        self, state_dict: dict[str, torch.Tensor], *, strict: bool
    ) -> None:
        self.loaded = (state_dict, strict)

    def to(self, device: torch.device) -> "_FakeStateModel":
        self.device = device
        return self

    def eval(self) -> "_FakeStateModel":
        self.evaluating = True
        return self


def _safe_checkpoint_payload() -> dict[str, object]:
    genes = [f"GENE{index:05d}" for index in range(state_inference.EXPECTED_SUPPORT_GENES)]
    return {
        "state_dict": {"weight": torch.tensor([1.0, 2.0])},
        "hyper_parameters": {
            "input_dim": state_inference.EXPECTED_SUPPORT_GENES,
            "hidden_dim": 672,
            "output_dim": state_inference.EXPECTED_SUPPORT_GENES,
            "pert_dim": state_inference.EXPECTED_PERT_DIM,
            "gene_names": genes,
            "cell_set_len": 128,
            "output_space": "all",
            "control_pert": "non-targeting",
            "embed_key": None,
            "batch_encoder": False,
            "predict_residual": True,
            "transformer_backbone_key": "llama",
            "transformer_backbone_kwargs": {"hidden_size": 672},
        },
    }


class SafeEsm2ConsumerTests(unittest.TestCase):
    def test_bayesian_loader_requires_explicit_hash_for_custom_map(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.pt"
            torch.save(
                {
                    "A": torch.arange(1, 5_121, dtype=torch.float32),
                    "B": torch.arange(2, 5_122, dtype=torch.float32),
                    "Q": torch.arange(3, 5_123, dtype=torch.float32),
                },
                path,
            )
            with self.assertRaisesRegex(
                ValueError, "requires --esm2-expected-sha256"
            ):
                bayesian.load_esm_features(path, ["A", "B"], ["Q"])

            public, query, missing, descriptor = bayesian.load_esm_features(
                path,
                ["A", "B"],
                ["Q"],
                expected_sha256=_sha256(path),
            )
            self.assertEqual(public.shape, (2, 5_120))
            self.assertEqual(query.shape, (1, 5_120))
            self.assertTrue(np.isfinite(public).all())
            self.assertTrue(np.isfinite(query).all())
            self.assertEqual(missing, [])
            self.assertEqual(descriptor["sha256"], _sha256(path))
            self.assertTrue(descriptor["restricted_weights_only_load"])
            self.assertTrue(descriptor["post_load_sha256_matches_preload"])

    def test_bayesian_loader_rejects_hash_mismatch_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "custom.pt"
            torch.save({"A": torch.ones(5_120)}, path)
            with mock.patch.object(authenticator.torch, "load") as loader:
                with self.assertRaisesRegex(ValueError, "SHA-256"):
                    bayesian.load_esm_features(
                        path,
                        ["A"],
                        ["A"],
                        expected_sha256="0" * 64,
                    )
                loader.assert_not_called()

    def test_legacy_state_map_loads_with_minimal_allowlist_and_exact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            torch.save(
                {
                    np.str_("non-targeting"): torch.zeros(5_120),
                    np.str_("A"): torch.ones(5_120),
                    np.str_("B"): torch.full((5_120,), 2.0),
                },
                path,
            )
            result, descriptor = state_inference.load_perturbation_embeddings(
                path,
                ["A", "B"],
                expected_sha256=_sha256(path),
                return_descriptor=True,
            )
            self.assertEqual(set(result), {"non-targeting", "A", "B"})
            self.assertEqual(tuple(result["A"].shape), (5_120,))
            self.assertEqual(descriptor["sha256"], _sha256(path))
            self.assertEqual(descriptor["path"], str(path.resolve()))
            self.assertEqual(descriptor["mtime_ns"], path.stat().st_mtime_ns)
            self.assertTrue(descriptor["restricted_weights_only_load"])
            safe_names = descriptor["explicit_safe_globals"]
            self.assertTrue(any(name.endswith(".scalar") for name in safe_names))
            self.assertTrue(any(name.endswith(".StrDType") for name in safe_names))

    def test_state_map_default_return_type_preserves_hash_pinned_callers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            torch.save(
                {
                    "non-targeting": torch.zeros(5_120),
                    "A": torch.ones(5_120),
                },
                path,
            )
            result = state_inference.load_perturbation_embeddings(
                path,
                ["A"],
                expected_sha256=_sha256(path),
            )
            self.assertIsInstance(result, dict)
            self.assertEqual(set(result), {"non-targeting", "A"})

    def test_state_map_requires_pre_registered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            torch.save(
                {
                    "non-targeting": torch.zeros(5_120),
                    "A": torch.ones(5_120),
                },
                path,
            )
            with self.assertRaisesRegex(TypeError, "expected_sha256"):
                state_inference.load_perturbation_embeddings(path, ["A"])

    def test_state_map_hash_mismatch_is_rejected_before_deserialization(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            torch.save(
                {
                    "non-targeting": torch.zeros(5_120),
                    "A": torch.ones(5_120),
                },
                path,
            )
            with mock.patch.object(authenticator.torch, "load") as loader:
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    state_inference.load_perturbation_embeddings(
                        path,
                        ["A"],
                        expected_sha256="0" * 64,
                    )
                loader.assert_not_called()

    def test_state_map_rejects_non_tensor_values_after_restricted_load(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pert_onehot_map.pt"
            torch.save(
                {
                    "non-targeting": torch.zeros(5_120),
                    "A": [1.0] * 5_120,
                },
                path,
            )
            with self.assertRaisesRegex(RuntimeError, "plain tensors"):
                state_inference.load_perturbation_embeddings(
                    path,
                    ["A"],
                    expected_sha256=_sha256(path),
                )

    def test_legacy_var_dims_pickle_is_never_deserialized(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "marker"
            legacy = root / "var_dims.pkl"
            with legacy.open("wb") as handle:
                pickle.dump(_Exploit(marker), handle)
            genes = ["A", "B"]
            safe = root / "version_0" / "hparams.yaml"
            safe.parent.mkdir()
            safe.write_text(
                yaml.safe_dump(
                    {
                        "gene_names": genes,
                        "input_dim": 2,
                        "output_dim": 2,
                        "pert_dim": state_inference.EXPECTED_PERT_DIM,
                    },
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            result, descriptor = state_inference.load_var_dims(
                legacy, genes, return_descriptor=True
            )
            self.assertEqual(result["gene_names"], genes)
            self.assertFalse(marker.exists())
            self.assertTrue(descriptor["legacy_pickle_requested"])
            self.assertFalse(descriptor["legacy_pickle_deserialized"])
            self.assertEqual(descriptor["sha256"], _sha256(safe))

    def test_legacy_var_dims_without_safe_sidecar_fails_actionably(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "marker"
            legacy = root / "var_dims.pkl"
            with legacy.open("wb") as handle:
                pickle.dump(_Exploit(marker), handle)
            with self.assertRaisesRegex(RuntimeError, "never deserialized"):
                state_inference.load_var_dims(legacy, ["A"])
            self.assertFalse(marker.exists())

    def test_var_dims_safe_yaml_rejects_unsafe_tags_and_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            unsafe = root / "dims.yaml"
            unsafe.write_text("gene_names: !!python/object/apply:os.system ['false']\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Invalid safe STATE"):
                state_inference.load_var_dims(unsafe, ["A"])

            duplicate = root / "duplicate.yaml"
            duplicate.write_text(
                "gene_names: [A]\ninput_dim: 1\ninput_dim: 1\noutput_dim: 1\npert_dim: 5120\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Duplicate safe metadata key"):
                state_inference.load_var_dims(duplicate, ["A"])

    def test_state_checkpoint_uses_restricted_payload_and_strict_state_dict(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "safe.ckpt"
            payload = _safe_checkpoint_payload()
            torch.save(payload, checkpoint)
            genes = payload["hyper_parameters"]["gene_names"]  # type: ignore[index]
            with mock.patch.object(
                state_inference, "_import_state_model_class", return_value=_FakeStateModel
            ):
                model, descriptor = state_inference.load_state_model(
                    checkpoint,
                    torch.device("cpu"),
                    expected_sha256=_sha256(checkpoint),
                    expected_gene_names=genes,  # type: ignore[arg-type]
                    return_descriptor=True,
                )
            self.assertIsInstance(model, _FakeStateModel)
            self.assertTrue(model.evaluating)
            self.assertEqual(model.device, torch.device("cpu"))
            self.assertIsNotNone(model.loaded)
            self.assertTrue(model.loaded[1])
            self.assertTrue(descriptor["restricted_weights_only_load"])
            self.assertEqual(descriptor["path"], str(checkpoint.resolve()))
            self.assertEqual(
                descriptor["mtime_ns"], checkpoint.stat().st_mtime_ns
            )
            self.assertFalse(descriptor["lightning_load_from_checkpoint_used"])
            self.assertTrue(descriptor["strict_state_dict_load"])

    def test_malicious_checkpoint_is_rejected_without_execution_or_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "marker"
            checkpoint = root / "malicious.ckpt"
            torch.save({"payload": _Exploit(marker)}, checkpoint)
            with mock.patch.object(state_inference, "_import_state_model_class") as importer:
                with self.assertRaisesRegex(RuntimeError, "Restricted weights-only"):
                    state_inference.load_state_model(
                        checkpoint,
                        torch.device("cpu"),
                        expected_sha256=_sha256(checkpoint),
                    )
                importer.assert_not_called()
            self.assertFalse(marker.exists())

    def test_state_checkpoint_requires_pre_registered_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "safe.ckpt"
            torch.save(_safe_checkpoint_payload(), checkpoint)
            with self.assertRaisesRegex(TypeError, "expected_sha256"):
                state_inference.load_state_model(checkpoint, torch.device("cpu"))

    def test_checkpoint_hash_mismatch_precedes_model_import(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "safe.ckpt"
            torch.save(_safe_checkpoint_payload(), checkpoint)
            with mock.patch.object(state_inference, "_import_state_model_class") as importer:
                with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                    state_inference.load_state_model(
                        checkpoint,
                        torch.device("cpu"),
                        expected_sha256="0" * 64,
                    )
                importer.assert_not_called()

    def test_checkpoint_symlink_is_not_normalized_around_safe_loader(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "safe.ckpt"
            torch.save(_safe_checkpoint_payload(), checkpoint)
            link = root / "linked.ckpt"
            link.symlink_to(checkpoint)
            resolved = state_inference.resolve_checkpoint(root, Path("linked.ckpt"))
            self.assertEqual(resolved, link)
            with mock.patch.object(state_inference, "_import_state_model_class") as importer:
                with self.assertRaisesRegex(RuntimeError, "Unable to open"):
                    state_inference.load_state_model(
                        resolved,
                        torch.device("cpu"),
                        expected_sha256=_sha256(checkpoint),
                    )
                importer.assert_not_called()

    def test_production_cli_requires_both_pre_registered_hashes(self) -> None:
        base = ["infer_state_effect_prior.py", "--model-dir", "/tmp/model"]
        with mock.patch.object(sys, "argv", base), self.assertRaises(SystemExit):
            state_inference.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            [*base, "--checkpoint-expected-sha256", "A" * 64],
        ), self.assertRaises(SystemExit):
            state_inference.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            [
                *base,
                "--checkpoint-expected-sha256",
                "A" * 64,
                "--perturbation-map-expected-sha256",
                "B" * 64,
            ],
        ):
            args = state_inference.parse_args()
        self.assertEqual(args.checkpoint_expected_sha256, "a" * 64)
        self.assertEqual(args.perturbation_map_expected_sha256, "b" * 64)

    def test_all_state_count_production_clis_require_both_hashes(self) -> None:
        cases = (
            (
                direct_counts.parse_args,
                ["generate_state_direct_counts.py", "--model-dir", "/tmp/model"],
            ),
            (
                scorer_counts.parse_args,
                ["generate_state_scorer_aware_counts.py", "--model-dir", "/tmp/model"],
            ),
            (
                native_counts.parse_args,
                [
                    "generate_native_state_anchor_counts.py",
                    "--controls-h5ad",
                    "/tmp/controls.h5ad",
                    "--panel-manifest",
                    "/tmp/panel.json",
                    "--model-dir",
                    "/tmp/model",
                    "--output-h5ad",
                    "/tmp/output.h5ad",
                    "--output-json",
                    "/tmp/output.json",
                ],
            ),
        )
        for parser, base in cases:
            with self.subTest(parser=parser.__module__), mock.patch.object(
                sys,
                "argv",
                base,
            ), self.assertRaises(SystemExit):
                parser()

            with self.subTest(parser=f"{parser.__module__}-accepted"), mock.patch.object(
                sys,
                "argv",
                [
                    *base,
                    "--checkpoint-expected-sha256",
                    "A" * 64,
                    "--perturbation-map-expected-sha256",
                    "B" * 64,
                ],
            ):
                args = parser()
            self.assertEqual(args.checkpoint_expected_sha256, "a" * 64)
            self.assertEqual(args.perturbation_map_expected_sha256, "b" * 64)

    def test_inference_source_has_no_unsafe_deserialization_escape_hatch(self) -> None:
        source = Path(state_inference.__file__).read_text(encoding="utf-8")
        self.assertNotIn("pickle.load", source)
        self.assertNotIn("weights_only=False", source)
        self.assertNotIn(".load_from_checkpoint(", source)

    def test_tracked_production_launchers_pin_regular_checkpoint_and_both_hashes(
        self,
    ) -> None:
        repository = Path(__file__).resolve().parents[1]
        invocation_counts = {
            "h100_infer_state_smoke_v0.sbatch": 1,
            "h100_infer_state_20k_best_v0.sbatch": 1,
            "h100_generate_state_direct_counts_smoke_v0.sbatch": 1,
            "h100_generate_state_direct_counts_v0.sbatch": 1,
            "h100_generate_public_state_anchor_v51_smoke.sbatch": 1,
            "h100_generate_public_state_anchor_v51_full.sbatch": 1,
            "h100_generate_public_candidate_v6.sbatch": 1,
            "h100_generate_cross_context_v2.sbatch": 1,
            "h100_generate_cross_context_v5_smoke.sbatch": 1,
            "h100_generate_state_anchor_v51_grid_smoke.sbatch": 1,
            "h100_generate_state_anchor_v51_smoke.sbatch": 2,
            "h100_generate_state_anchor_v51_final_smoke.sbatch": 2,
        }
        sha_pattern = re.compile(r'^[a-f0-9]{64}$')
        for filename, invocations in invocation_counts.items():
            with self.subTest(filename=filename):
                text = (repository / "slurm" / filename).read_text(encoding="utf-8")
                self.assertNotRegex(
                    text,
                    r"--checkpoint(?:\s+|=)(?:checkpoints/)?selected\.ckpt(?:\s|$)",
                )
                self.assertEqual(
                    text.count("--checkpoint-expected-sha256"), invocations
                )
                self.assertEqual(
                    text.count("--perturbation-map-expected-sha256"), invocations
                )
                assignments = dict(
                    re.findall(
                        r'^readonly (CHECKPOINT_EXPECTED_SHA256|PERTURBATION_MAP_EXPECTED_SHA256)="([a-f0-9]{64})"$',
                        text,
                        flags=re.MULTILINE,
                    )
                )
                self.assertEqual(set(assignments), {
                    "CHECKPOINT_EXPECTED_SHA256",
                    "PERTURBATION_MAP_EXPECTED_SHA256",
                })
                self.assertTrue(all(sha_pattern.fullmatch(value) for value in assignments.values()))


if __name__ == "__main__":
    unittest.main()
