"""Focused tests for the native-axis sealed-validation STATE adapter."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from generate_native_state_anchor_counts import (  # noqa: E402
    DATA_SCHEMA,
    CONTROL_ROLE,
    FrozenPanel,
    build_gene_maps,
    combine_support_delta,
    load_frozen_panel,
    load_residual_overlay,
    render_exact_native_counts,
    sample_control_indices,
    validate_native_candidate,
    validate_controls_contract,
)
from infer_state_effect_prior import describe_file  # noqa: E402


def synthetic_panel(root: Path) -> FrozenPanel:
    manifest_path = root / "panel.json"
    csv_path = root / "panel.csv"
    return FrozenPanel(
        targets=("T1", "T2", "T3"),
        routes=("direct", "held_target", "direct"),
        manifest={},
        manifest_path=manifest_path,
        csv_path=csv_path,
    )


class FrozenPanelTests(unittest.TestCase):
    def test_manifest_authenticates_target_csv_and_routes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_path = root / "panel.csv"
            manifest_path = root / "panel.json"
            pd.DataFrame(
                {
                    "target_gene": ["T1", "T2", "T3"],
                    "validation_route": ["direct", "held_target", "direct"],
                }
            ).to_csv(csv_path, index=False)
            manifest_path.write_text(
                json.dumps(
                    {
                        "schema": "vcc-public-validation-manifest-v1",
                        "selection_contract": {
                            "effect_values_accessed": False,
                            "panel_targets": 3,
                            "direct_targets": 2,
                            "held_targets": 1,
                        },
                        "provenance": {"output_csv": describe_file(csv_path)},
                    }
                ),
                encoding="utf-8",
            )
            panel = load_frozen_panel(manifest_path)
            self.assertEqual(panel.targets, ("T1", "T2", "T3"))
            self.assertEqual(panel.routes, ("direct", "held_target", "direct"))

            csv_path.write_text("target_gene,validation_route\nT1,direct\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "size changed|hash changed"):
                load_frozen_panel(manifest_path)


class ControlsFirewallTests(unittest.TestCase):
    @staticmethod
    def controls() -> ad.AnnData:
        data = ad.AnnData(
            X=sp.csr_matrix(np.asarray([[5, 2, 0], [3, 0, 4]], dtype=np.int32)),
            obs=pd.DataFrame(
                {
                    "context": ["HepG2", "HepG2"],
                    "target_gene": ["non-targeting", "non-targeting"],
                },
                index=["c0", "c1"],
            ),
            var=pd.DataFrame(
                {"is_state_support": [True, False, True]},
                index=pd.Index(["g0", "g1", "g2"], name="gene_name"),
            ),
        )
        data.uns["public_validation"] = {
            "schema": DATA_SCHEMA,
            "role": CONTROL_ROLE,
            "sealed_treated_profiles_present": False,
        }
        return data

    def test_only_controls_role_and_rows_are_accepted(self) -> None:
        controls = self.controls()
        context, genes = validate_controls_contract(
            controls, expected_context="HepG2", support_genes=["g0", "g2"]
        )
        self.assertEqual(context, "HepG2")
        self.assertEqual(genes, ["g0", "g1", "g2"])

        controls.obs.loc["c1", "target_gene"] = "T1"
        with self.assertRaisesRegex(RuntimeError, "non-control rows"):
            validate_controls_contract(controls, support_genes=["g0", "g2"])

    def test_sealed_role_is_rejected_before_expression_is_used(self) -> None:
        controls = self.controls()
        controls.uns["public_validation"]["role"] = "sealed-scorer-input"
        controls.uns["public_validation"]["sealed_treated_profiles_present"] = True
        with self.assertRaisesRegex(RuntimeError, "not a generator controls input"):
            validate_controls_contract(controls)


class GeneAndResidualTests(unittest.TestCase):
    def test_native_support_mapping_is_bidirectional(self) -> None:
        maps = build_gene_maps(
            ["g2", "native-only", "g0", "T1"],
            ["g0", "g1", "g2", "T1"],
        )
        np.testing.assert_array_equal(maps.support_to_native, [2, -1, 0, 3])
        np.testing.assert_array_equal(maps.native_to_support, [2, -1, 0, 3])

    def test_residual_aligns_axes_and_keeps_fallback_rows_zero(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "residual.npz"
            # Source target order is T3, T1, T2 and source gene order is g2, g0, g1.
            effects = np.asarray(
                [
                    [3.0, 1.0, 0.0],
                    [4.0, 2.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float32,
            )
            np.savez_compressed(
                path,
                effects=effects[None, :, :],
                target_names=np.asarray(["T3", "T1", "T2"]),
                gene_names=np.asarray(["g2", "g0", "g1"]),
                contexts=np.asarray(["HepG2"]),
                direct_mask=np.asarray([1, 1, 0], dtype=np.uint8),
                fallback_mask=np.asarray([0, 0, 1], dtype=np.uint8),
            )
            overlay = load_residual_overlay(
                path,
                alpha=0.25,
                panel=synthetic_panel(root),
                native_genes=["g0", "g1", "g2", "T1", "T2", "T3"],
                support_genes=["g2", "T3", "g0", "T1", "T2"],
                context="HepG2",
            )
            self.assertTrue(overlay.qc["artifact_read"])
            np.testing.assert_array_equal(overlay.direct_mask, [True, False, True])
            # T1 aligns to support positions g2=4 and g0=2; g1 is native-only and zero.
            np.testing.assert_allclose(overlay.effects[0], [4.0, 0.0, 2.0, 0.0, 0.0])
            np.testing.assert_array_equal(overlay.effects[1], np.zeros(5))
            np.testing.assert_allclose(overlay.effects[2], [3.0, 0.0, 1.0, 0.0, 0.0])
            with self.assertRaisesRegex(RuntimeError, "Residual context"):
                load_residual_overlay(
                    path,
                    alpha=0.25,
                    panel=synthetic_panel(root),
                    native_genes=["g0", "g1", "g2", "T1", "T2", "T3"],
                    support_genes=["g2", "T3", "g0", "T1", "T2"],
                    context="Jurkat",
                )

    def test_residual_rejects_fallback_signal_and_native_only_signal(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "residual.npz"
            base = dict(
                target_names=np.asarray(["T1", "T2", "T3"]),
                gene_names=np.asarray(["g0", "native-only"]),
                direct_mask=np.asarray([1, 0, 1], dtype=np.uint8),
                fallback_mask=np.asarray([0, 1, 0], dtype=np.uint8),
            )
            effects = np.zeros((3, 2), dtype=np.float32)
            effects[1, 0] = 0.1
            np.savez_compressed(path, effects=effects, **base)
            with self.assertRaisesRegex(RuntimeError, "fallback residual rows"):
                load_residual_overlay(
                    path,
                    alpha=1.0,
                    panel=synthetic_panel(root),
                    native_genes=["g0", "native-only", "T1", "T2", "T3"],
                    support_genes=["g0", "T1", "T2", "T3"],
                )

            effects[1, 0] = 0.0
            effects[0, 1] = 0.2
            np.savez_compressed(path, effects=effects, **base)
            with self.assertRaisesRegex(RuntimeError, "native non-STATE gene"):
                load_residual_overlay(
                    path,
                    alpha=1.0,
                    panel=synthetic_panel(root),
                    native_genes=["g0", "native-only", "T1", "T2", "T3"],
                    support_genes=["g0", "T1", "T2", "T3"],
                )

    def test_alpha_zero_does_not_open_declared_residual(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "does_not_exist.npz"
            overlay = load_residual_overlay(
                missing,
                alpha=0.0,
                panel=synthetic_panel(root),
                native_genes=["g0"],
                support_genes=["g0"],
            )
            self.assertIsNone(overlay.effects)
            self.assertFalse(overlay.qc["artifact_read"])


class EffectAndCountTests(unittest.TestCase):
    def test_anchor_then_residual_then_clip_then_target_force(self) -> None:
        state = np.asarray([[1.0, -2.0, 0.0]], dtype=np.float32)
        residual = np.asarray([0.8, 0.4, -0.5], dtype=np.float32)
        combined, qc = combine_support_delta(
            state,
            residual,
            state_weight=1.0,
            state_clip=0.5,
            state_bounding="clip",
            residual_alpha=0.25,
            combined_clip=0.6,
            target_support_index=1,
            target_policy="force",
            target_remaining_fraction=0.2,
        )
        np.testing.assert_allclose(combined[0, [0, 2]], [0.6, -0.125])
        self.assertAlmostEqual(float(combined[0, 1]), float(np.log(0.2)), places=6)
        self.assertEqual(qc["state_outside_bound"], 2)
        self.assertEqual(qc["combined_outside_bound"], 1)

    def test_state_weight_is_applied_before_tanh_bounding(self) -> None:
        state = np.asarray([[2.0, -2.0]], dtype=np.float32)
        combined, _ = combine_support_delta(
            state,
            np.zeros(2, dtype=np.float32),
            state_weight=0.5,
            state_clip=0.5,
            state_bounding="tanh",
            residual_alpha=0.0,
            combined_clip=1.0,
            target_support_index=0,
            target_policy="off",
            target_remaining_fraction=0.2,
        )
        expected = 0.5 * np.tanh((0.5 * state) / 0.5)
        np.testing.assert_allclose(combined, expected, rtol=1e-6, atol=1e-7)
        self.assertGreater(float(combined[0, 0]), 0.45)

    def test_exact_renderer_preserves_library_and_native_only_counts(self) -> None:
        base = sp.csr_matrix(np.asarray([[5, 3, 7, 2]], dtype=np.int32))
        emitted, qc = render_exact_native_counts(
            base,
            np.zeros((1, 3), dtype=np.float32),
            np.asarray([0, 1, -1, 2], dtype=np.int64),
            max_genes_per_cell=3,
            integerization="largest-remainder",
            rng=np.random.default_rng(7),
        )
        np.testing.assert_array_equal(emitted.toarray(), np.asarray([[6, 4, 7, 0]]))
        self.assertEqual(int(emitted.sum()), int(base.sum()))
        self.assertEqual(int(emitted[0, 2]), 7)
        self.assertEqual(qc["native_non_support_count_mismatches"], 0)
        self.assertEqual(qc["redistributed_shared_counts"], 2)

    def test_sampling_is_seeded_and_uniform_without_stratum_column(self) -> None:
        obs = pd.DataFrame(index=[f"c{index}" for index in range(10)])
        first, allocation = sample_control_indices(
            obs, 4, np.random.default_rng(42), strata_column="ntc_id"
        )
        second, _ = sample_control_indices(
            obs, 4, np.random.default_rng(42), strata_column="ntc_id"
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(allocation, {"__uniform__": 4})
        self.assertEqual(len(set(first.tolist())), 4)

    def test_backed_candidate_validator_accepts_native_csr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "candidate.h5ad"
            candidate = ad.AnnData(
                X=sp.csr_matrix(np.asarray([[2, 1], [3, 0]], dtype=np.int32)),
                obs=pd.DataFrame(
                    {
                        "context": ["HepG2", "HepG2"],
                        "target_gene": ["T1", "T1"],
                        "source_control_row": [0, 1],
                    },
                    index=["p0", "p1"],
                ),
                var=pd.DataFrame(index=pd.Index(["g0", "T1"], name="gene_name")),
            )
            candidate.write_h5ad(path)
            report = validate_native_candidate(
                path,
                genes=["g0", "T1"],
                context="HepG2",
                targets=["T1"],
                cells_per_group=2,
                expected_nnz=3,
            )
            self.assertTrue(all(report["checks"].values()))


if __name__ == "__main__":
    unittest.main()
