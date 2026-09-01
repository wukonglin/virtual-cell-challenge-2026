"""Synthetic tests for the leakage-safe P3 count-identity comparator."""

from __future__ import annotations

import copy
import json
import sys
import subprocess
import tempfile
import unittest
from argparse import Namespace
from dataclasses import replace
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import compare_public_p3_count_identity as module  # noqa: E402


class PublicP3CountIdentityTests(unittest.TestCase):
    def _build_bundle(
        self,
        root: Path,
        *,
        held_difference: bool = False,
        source_pairing_difference: bool = False,
    ) -> Namespace:
        targets = ["D0", "D1", "H0", "H1"]
        routes = ["direct", "direct", "held_target", "held_target"]
        cells_per_group = 2
        genes = ["G0", "G1", "G2"]
        manifest = root / "panel.csv"
        pd.DataFrame(
            {"target_gene": targets, "validation_route": routes}
        ).to_csv(manifest, index=False)

        shared_paths: dict[str, Path] = {}
        for name in module.REQUIRED_SPEC_INPUTS:
            if name == "panel_csv":
                shared_paths[name] = manifest
                continue
            suffix = ".json" if name.endswith("json") or name == "panel_manifest" else ".bin"
            path = root / f"{name}{suffix}"
            path.write_bytes(f"safe synthetic {name}".encode("utf-8"))
            shared_paths[name] = path
        input_descriptors = {
            name: module.describe_file(path) for name, path in shared_paths.items()
        }
        state_source_dir = root / "state_source"
        (state_source_dir / "src").mkdir(parents=True)
        (state_source_dir / "src" / "runtime.py").write_text(
            "# synthetic STATE runtime\n", encoding="utf-8"
        )
        for command in (
            ("git", "init", "-q"),
            ("git", "config", "user.email", "test@example.invalid"),
            ("git", "config", "user.name", "Contract Test"),
            ("git", "add", "src/runtime.py"),
            ("git", "commit", "-q", "-m", "runtime"),
        ):
            subprocess.run(command, cwd=state_source_dir, check=True)
        state_source = module.authenticate_state_source(state_source_dir)

        obs = pd.DataFrame(
            {
                "context": ["HepG2"] * 8,
                "target_gene": np.repeat(targets, cells_per_group),
                "source_control_row": np.tile([2, 5], 4),
                "source_control_cell_id": np.tile(["c2", "c5"], 4),
            },
            index=[f"cell-{index}" for index in range(8)],
        )
        var = pd.DataFrame(
            {"is_state_support": [True, True, False]}, index=genes
        )
        left_dense = np.asarray(
            [
                [4, 1, 0],
                [2, 0, 3],
                [3, 1, 1],
                [1, 2, 2],
                [5, 0, 0],
                [0, 2, 3],
                [2, 1, 2],
                [1, 3, 1],
            ],
            dtype=np.int32,
        )
        right_dense = left_dense.copy()
        right_dense[0] = [3, 2, 0]
        right_dense[3] = [1, 1, 3]
        if held_difference:
            right_dense[4] = [4, 1, 0]
        right_obs = obs.copy()
        if source_pairing_difference:
            right_obs.loc["cell-0", "source_control_row"] = 7

        def write_candidate(side: str, alpha: float, dense: np.ndarray, side_obs: pd.DataFrame) -> tuple[Path, Path, Path]:
            prediction = root / f"{side}_prediction.h5ad"
            ad.AnnData(sp.csr_matrix(dense), obs=side_obs, var=var).write_h5ad(prediction)
            configuration = {
                "context": "HepG2",
                "cells_per_group": cells_per_group,
                "residual_alpha": alpha,
                "seed": 20260901,
                "target_policy": "force",
                "target_remaining_fraction": 0.40,
            }
            spec = {
                "schema": module.SPEC_SCHEMA,
                "output_tag": f"p3_{side}",
                "context": "HepG2",
                "context_prefix": "hepg2",
                "configuration": configuration,
                "axes": {
                    "targets": 4,
                    "direct_targets": 2,
                    "held_targets": 2,
                    "native_genes": 3,
                    "control_cells": 10,
                    "state_support_genes": 2,
                },
                "inputs": input_descriptors,
                "state_source": state_source,
                "checkpoint_selection": {"global_step": 1, "val_loss": 1.0},
                "residual_contract": {
                    "schema": "synthetic",
                    "held_target_rows_are_zero": True,
                    "recipient_treated_profiles_used": False,
                    "shared_response_weight": 0.0,
                },
                "firewall": {
                    "generator_inputs_include_sealed_truth": False,
                    "controls_role": "controls-only-generator-input",
                    "treated_profiles_read_while_planning": False,
                },
                "provenance_contract": {
                    "schema": module.STRICT_PROVENANCE_CONTRACT,
                    "mode": "strict",
                    "generator_report_provenance_authenticated": True,
                    "optional_state_config_bound": True,
                },
            }
            spec_path = root / f"{side}_spec.json"
            spec_path.write_text(json.dumps(spec), encoding="utf-8")
            provenance = {
                "state_source_commit": state_source["repository_commit"],
                "controls_h5ad": input_descriptors["controls_h5ad"],
                "panel_manifest": input_descriptors["panel_manifest"],
                "panel_csv": input_descriptors["panel_csv"],
                "support_gene_axis": input_descriptors["support_genes"],
                "checkpoint": input_descriptors["checkpoint"],
                "checkpoint_selection": input_descriptors["selection_json"],
                "script": input_descriptors["generator_script"],
                "generate_state_direct_counts_helper": input_descriptors[
                    "generate_state_direct_counts_helper"
                ],
                "infer_state_effect_prior_helper": input_descriptors[
                    "infer_state_effect_prior_helper"
                ],
                "perturbation_map": input_descriptors["perturbation_map"],
                "var_dims": input_descriptors["var_dims"],
                "state_config": input_descriptors["state_config"],
                "residual_artifact": input_descriptors["residual_npz"] if alpha else None,
                "output_h5ad": module.describe_file(prediction),
            }
            generation = {
                "schema": module.PREDICTION_SCHEMA,
                "artifact_type": "sealed_public_validation_prediction",
                "full_frozen_panel_contract": True,
                "data_firewall": {
                    "sealed_treated_profiles_read": False,
                    "truth_inputs_read": [],
                },
                "configuration": configuration,
                "validation": {"shape": [8, 3], "groups": 4, "failed_checks": []},
                "scientific_qc": {
                    "all_libraries_exact": True,
                    "all_native_non_support_counts_exact": True,
                    "target_knockdown_failures": [],
                },
                "residual": {
                    "enabled": bool(alpha),
                    "alpha": alpha,
                    "artifact_read": bool(alpha),
                    "declared_path": str(shared_paths["residual_npz"]),
                },
                "provenance": provenance,
            }
            generation_path = root / f"{side}_generation.json"
            generation_path.write_text(json.dumps(generation), encoding="utf-8")
            verification = {
                "schema": module.VERIFICATION_SCHEMA,
                "status": "passed",
                "output_tag": spec["output_tag"],
                "context": "HepG2",
                "configuration": configuration,
                "validation_mode": "strict-v2",
                "provenance": {
                    "spec": module.describe_file(spec_path),
                    "generation_json": module.describe_file(generation_path),
                    "prediction_h5ad": module.describe_file(prediction),
                    "authenticated_state_source": state_source,
                },
                "checks": {name: True for name in module.REQUIRED_VERIFICATION_CHECKS},
            }
            verification_path = root / f"{side}_verified.json"
            verification_path.write_text(json.dumps(verification), encoding="utf-8")
            return spec_path, verification_path, prediction

        left_spec, left_verification, left_prediction = write_candidate(
            "state", 0.0, left_dense, obs
        )
        right_spec, right_verification, right_prediction = write_candidate(
            "residual", 0.1, right_dense, right_obs
        )
        return Namespace(
            manifest=manifest,
            left_spec=left_spec,
            left_verification=left_verification,
            left_prediction=left_prediction,
            right_spec=right_spec,
            right_verification=right_verification,
            right_prediction=right_prediction,
            chunk_rows=1,
            output_json=None,
        )

    def test_authenticated_pair_passes_with_exact_held_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            payload = module.compare_candidates(self._build_bundle(Path(directory)))
            self.assertEqual(payload["status"], "passed")
            self.assertTrue(
                payload["count_comparison"]["held_target"]["exact_count_identity"]
            )
            self.assertEqual(
                payload["count_comparison"]["held_target"]["differing_rows"], 0
            )
            direct = payload["count_comparison"]["direct"]
            self.assertEqual(direct["differing_rows"], 2)
            self.assertEqual(direct["differing_entries"], 4)
            self.assertNotEqual(
                direct["left_logical_csr_sha256"], direct["right_logical_csr_sha256"]
            )

    def test_any_held_count_difference_is_a_hard_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(Path(directory), held_difference=True)
            with self.assertRaisesRegex(RuntimeError, "Held-target count matrices differ"):
                module.compare_candidates(args)

    def test_source_control_pairing_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(
                Path(directory), source_pairing_difference=True
            )
            with self.assertRaisesRegex(RuntimeError, "Observation annotations differ"):
                module.compare_candidates(args)

    def test_prediction_tampering_is_detected_before_matrix_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(Path(directory))
            with args.right_prediction.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RuntimeError, "prediction size differs"):
                module.compare_candidates(args)

    def test_only_residual_alpha_may_differ(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(Path(directory))
            cache: dict[tuple[str, int, str], dict[str, object]] = {}
            left = module.authenticate_candidate(
                spec_path=args.left_spec,
                verification_path=args.left_verification,
                prediction_path=args.left_prediction,
                cache=cache,
                label="left",
            )
            right = module.authenticate_candidate(
                spec_path=args.right_spec,
                verification_path=args.right_verification,
                prediction_path=args.right_prediction,
                cache=cache,
                label="right",
            )
            changed_spec = copy.deepcopy(right.spec)
            changed_spec["configuration"]["seed"] += 1
            changed = replace(right, spec=changed_spec)
            with self.assertRaisesRegex(RuntimeError, "differ only in residual_alpha"):
                module.validate_pair_contract(left, changed)

    def test_count_audit_is_locked_to_the_exact_c_d_factorial_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(Path(directory))
            cache: dict[tuple[str, int, str], dict[str, object]] = {}
            left = module.authenticate_candidate(
                spec_path=args.left_spec,
                verification_path=args.left_verification,
                prediction_path=args.left_prediction,
                cache=cache,
                label="left",
            )
            right = module.authenticate_candidate(
                spec_path=args.right_spec,
                verification_path=args.right_verification,
                prediction_path=args.right_prediction,
                cache=cache,
                label="right",
            )

            wrong_left_spec = copy.deepcopy(left.spec)
            wrong_right_spec = copy.deepcopy(right.spec)
            wrong_left_spec["configuration"]["target_remaining_fraction"] = 0.20
            wrong_right_spec["configuration"]["target_remaining_fraction"] = 0.20
            with self.assertRaisesRegex(RuntimeError, "requires target_remaining_fraction"):
                module.validate_pair_contract(
                    replace(left, spec=wrong_left_spec),
                    replace(right, spec=wrong_right_spec),
                )

            wrong_alpha_spec = copy.deepcopy(right.spec)
            wrong_alpha_spec["configuration"]["residual_alpha"] = 0.20
            wrong_alpha = replace(right, spec=wrong_alpha_spec)
            with self.assertRaisesRegex(RuntimeError, "requires residual_alpha"):
                module.validate_pair_contract(left, wrong_alpha)

    def test_logical_hashes_do_not_depend_on_chunk_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self._build_bundle(Path(directory))
            left = ad.read_h5ad(args.left_prediction, backed="r")
            right = ad.read_h5ad(args.right_prediction, backed="r")
            try:
                mask = np.asarray([True, True, False, False])
                one = module.stream_compare_counts(
                    left, right, direct_mask=mask, cells_per_group=2, chunk_rows=1
                )
                two = module.stream_compare_counts(
                    left, right, direct_mask=mask, cells_per_group=2, chunk_rows=2
                )
            finally:
                left.file.close()
                right.file.close()
            self.assertEqual(one, two)


if __name__ == "__main__":
    unittest.main()
