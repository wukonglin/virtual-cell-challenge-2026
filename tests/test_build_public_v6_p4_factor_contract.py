"""Focused tests for the strict P4 STATE-gamma factor contract."""

from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import build_public_v6_p4_factor_contract as module  # noqa: E402


class PublicV6P4FactorContractTests(unittest.TestCase):
    @staticmethod
    def _descriptor(path: str, marker: str) -> dict[str, object]:
        return {
            "path": str(Path("/tmp") / path),
            "size_bytes": 10,
            "sha256": marker * 64,
        }

    def _candidate(self, arm: str) -> module.Candidate:
        definition = module.ARM_DEFINITIONS[arm]
        pinned_inputs = {
            name: {
                "path": str(Path("/tmp") / f"pinned_{name}.bin"),
                "size_bytes": 10,
                "sha256": digest,
            }
            for name, digest in module.PINNED_V51_INPUT_SHA256.items()
        }
        specification = {
            "schema": module.SPEC_SCHEMA,
            "context": "HepG2",
            "context_prefix": "hepg2",
            "output_tag": definition["output_tag"],
            "axes": {
                "targets": 300,
                "direct_targets": 267,
                "held_targets": 33,
            },
            "checkpoint_selection": {"global_step": 16000, "val_loss": 1.6},
            "configuration": {
                "context": "HepG2",
                "cells_per_group": 400,
                "combined_effect_clip": 0.65,
                "residual_alpha": 0.10,
                "seed": 20260901,
                "state_effect_weight": definition["state_effect_weight"],
                "target_remaining_fraction": 0.20,
            },
            "inputs": pinned_inputs,
            "residual_contract": {
                "schema": "vcc-public-state-residual-v1",
                "held_target_rows_are_zero": True,
                "recipient_treated_profiles_used": False,
                "shared_response_weight": 0.0,
            },
            "firewall": {
                "generator_inputs_include_sealed_truth": False,
                "treated_profiles_read_while_planning": False,
            },
            "provenance_contract": {
                "schema": "vcc-public-generator-provenance-v2",
                "mode": "strict",
            },
            "state_source": {
                **module.PINNED_STATE_SOURCE,
            },
        }
        descriptors = {
            "spec": self._descriptor(f"{arm}_spec.json", "b"),
            "generation_verified": self._descriptor(
                f"{arm}_generation_verified.json", "c"
            ),
            "generation_json": self._descriptor(f"{arm}_generation.json", "d"),
            "prediction_h5ad": self._descriptor(f"{arm}_prediction.h5ad", "e"),
        }
        return module.Candidate(
            spec_path=Path(descriptors["spec"]["path"]),
            verification_path=Path(descriptors["generation_verified"]["path"]),
            prediction_path=Path(descriptors["prediction_h5ad"]["path"]),
            generation_path=Path(descriptors["generation_json"]["path"]),
            spec=specification,
            verification={},
            generation={},
            descriptors=descriptors,
        )

    def _candidates(self) -> dict[str, module.Candidate]:
        return {arm: self._candidate(arm) for arm in module.ARM_KEYS}

    @staticmethod
    def _realized() -> dict[str, object]:
        return {
            "shared_generation_identity_sha256": "f" * 64,
            "pairwise": {
                pair: {"differing_entries": 1} for pair in module.PAIR_KEYS
            }
        }

    def test_exact_preregistered_factor_contract_passes(self) -> None:
        candidates = self._candidates()
        factor = module.validate_factor_candidates(candidates)
        self.assertEqual(factor["sole_configuration_difference"], "state_effect_weight")
        self.assertEqual(
            factor["levels_by_arm"],
            {"g100": 1.0, "g075": 0.75, "g050": 0.5},
        )
        self.assertEqual(len(factor["shared_spec_identity_sha256"]), 64)

        payload = module.build_factor_payload(candidates, self._realized())
        self.assertEqual(payload["status"], "passed")
        self.assertTrue(all(payload["checks"].values()))

    def test_nonfactor_configuration_drift_is_rejected(self) -> None:
        candidates = self._candidates()
        candidates["g050"].spec["configuration"]["combined_effect_clip"] = 0.60
        with self.assertRaisesRegex(RuntimeError, "outside state_effect_weight"):
            module.validate_factor_candidates(candidates)

    def test_generator_input_drift_is_rejected(self) -> None:
        candidates = self._candidates()
        candidates["g075"].spec["inputs"]["residual_npz"] = self._descriptor(
            "different.bin", "f"
        )
        with self.assertRaisesRegex(RuntimeError, "generator input differs"):
            module.validate_factor_candidates(candidates)

    def test_shared_substitution_of_frozen_v51_residual_is_rejected(self) -> None:
        candidates = self._candidates()
        for candidate in candidates.values():
            candidate.spec["inputs"]["residual_npz"] = self._descriptor(
                "substituted_residual.npz", "f"
            )
        with self.assertRaisesRegex(RuntimeError, "frozen V5.1 artifact"):
            module.validate_factor_candidates(candidates)

    def test_tag_and_level_are_preregistered(self) -> None:
        candidates = self._candidates()
        wrong_tag = copy.deepcopy(candidates)
        wrong_tag["g075"].spec["output_tag"] = "renamed"
        with self.assertRaisesRegex(RuntimeError, "output tag"):
            module.validate_factor_candidates(wrong_tag)

        candidates = self._candidates()
        candidates["g075"].spec["configuration"]["state_effect_weight"] = 0.60
        with self.assertRaisesRegex(RuntimeError, "STATE effect weight"):
            module.validate_factor_candidates(candidates)

    def test_comparison_binding_requires_receipt_specs(self) -> None:
        candidates = self._candidates()
        receipt = module.build_factor_payload(candidates, self._realized())
        binding = module.bind_factor_comparison(
            receipt,
            baseline_label="p4_g100_p0_a010",
            baseline_spec=candidates["g100"].spec_path,
            candidate_label="p4_g075_p0_a010",
            candidate_spec=candidates["g075"].spec_path,
        )
        self.assertEqual(binding["baseline_level"], 1.0)
        self.assertEqual(binding["candidate_level"], 0.75)

        with self.assertRaisesRegex(RuntimeError, "absent"):
            module.bind_factor_comparison(
                receipt,
                baseline_label="v51_alpha010",
                baseline_spec=Path("/tmp/legacy.json"),
                candidate_label="p4_g075_p0_a010",
                candidate_spec=candidates["g075"].spec_path,
            )

    def test_receipt_validation_rehashes_bound_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidates = self._candidates()
            for arm, candidate in tuple(candidates.items()):
                descriptors: dict[str, dict[str, object]] = {}
                for name in module.PRIMARY_DESCRIPTOR_KEYS:
                    path = root / f"{arm}_{name}.bin"
                    path.write_bytes(f"{arm}:{name}".encode("ascii"))
                    descriptors[name] = module.describe_file(path)
                candidates[arm] = replace(
                    candidate,
                    spec_path=Path(descriptors["spec"]["path"]),
                    verification_path=Path(
                        descriptors["generation_verified"]["path"]
                    ),
                    generation_path=Path(descriptors["generation_json"]["path"]),
                    prediction_path=Path(descriptors["prediction_h5ad"]["path"]),
                    descriptors=descriptors,
                )

            receipt = module.build_factor_payload(candidates, self._realized())
            receipt_path = root / "factor.json"
            receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
            with mock.patch.object(
                module,
                "authenticate_candidate",
                side_effect=[candidates[arm] for arm in module.ARM_KEYS],
            ), mock.patch.object(
                module,
                "validate_realized_predictions",
                return_value=self._realized(),
            ):
                self.assertEqual(module.validate_factor_receipt(receipt_path), receipt)

            tampered = Path(
                candidates["g075"].descriptors["generation_verified"]["path"]
            )
            tampered.write_bytes(b"tampered")
            with mock.patch.object(
                module,
                "authenticate_candidate",
                side_effect=[candidates["g100"]],
            ), mock.patch.object(
                module,
                "validate_realized_predictions",
                return_value=self._realized(),
            ), self.assertRaisesRegex(RuntimeError, "size differs"):
                module.validate_factor_receipt(receipt_path)

    def test_realized_predictions_require_matched_pairing_and_distinct_counts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = ("D0", "D1", "H0", "H1")
            routes = ("direct", "direct", "held_target", "held_target")
            obs = pd.DataFrame(
                {
                    "context": ["HepG2"] * 8,
                    "target_gene": np.repeat(targets, 2),
                    "source_control_row": np.tile([1, 3], 4),
                    "source_control_cell_id": np.tile(["c1", "c3"], 4),
                },
                index=[f"cell-{index}" for index in range(8)],
            )
            var = pd.DataFrame(index=["G0", "G1", "G2"])
            base = np.asarray(
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
            matrices = {
                "g100": base,
                "g075": base.copy(),
                "g050": base.copy(),
            }
            matrices["g075"][[0, 4]] = [[3, 2, 0], [4, 1, 0]]
            matrices["g050"][[1, 5]] = [[1, 1, 3], [0, 1, 4]]

            groups = []
            for ordinal, (target, route) in enumerate(zip(targets, routes)):
                groups.append(
                    {
                        "cells": 2,
                        "chunk_sizes": [2],
                        "context": "HepG2",
                        "control_strata": 1,
                        "control_strata_used": 1,
                        "count_seed": ordinal + 10,
                        "delta_definition": "paired",
                        "effect_values": 6,
                        "input_library": {"mean": 5.0},
                        "input_normalization": "log1p",
                        "output_library": {"mean": 5.0},
                        "paired_passes": 1,
                        "panel_ordinal": ordinal,
                        "prediction_ranges": [{"paired_delta_abs_max": 1.0}],
                        "raw_state_delta": {"mean": 0.0},
                        "residual_nonzero": 0,
                        "sample_seed": ordinal + 20,
                        "source_control_index_sha256": str(ordinal) * 64,
                        "target_gene": target,
                        "target_modeled_delta_chunk_means": {"mean": 0.0},
                        "target_ordinal": ordinal,
                        "target_sum_before": 5,
                        "validation_route": route,
                    }
                )

            candidates = self._candidates()
            modeled_target_means = {"g100": 0.30, "g075": 0.20, "g050": 0.10}
            for arm in module.ARM_KEYS:
                prediction = root / f"{arm}.h5ad"
                ad.AnnData(sp.csr_matrix(matrices[arm]), obs=obs, var=var).write_h5ad(
                    prediction
                )
                spec = copy.deepcopy(candidates[arm].spec)
                spec["configuration"]["cells_per_group"] = 2
                arm_groups = copy.deepcopy(groups)
                for group in arm_groups:
                    group["target_modeled_delta_chunk_means"] = {
                        "mean": modeled_target_means[arm]
                    }
                generation = {
                    "axes": {"targets": 4, "genes": 3},
                    "checkpoint_selection": {"global_step": 1},
                    "data_firewall": {"sealed_treated_profiles_read": False},
                    "model": {"class": "synthetic"},
                    "residual": {"alpha": 0.1},
                    "validation": {
                        "checks": {"shape_exact": True},
                        "data_dtype": "int32",
                        "failed_checks": [],
                        "groups": 4,
                        "shape": [8, 3],
                    },
                    "qc": {"groups": arm_groups},
                }
                descriptors = dict(candidates[arm].descriptors)
                descriptors["prediction_h5ad"] = module.describe_file(prediction)
                candidates[arm] = replace(
                    candidates[arm],
                    prediction_path=prediction,
                    spec=spec,
                    generation=generation,
                    descriptors=descriptors,
                )

            realized = module.validate_realized_predictions(candidates, chunk_rows=1)
            self.assertEqual(set(realized["pairwise"]), set(module.PAIR_KEYS))
            self.assertTrue(
                all(
                    pair["differing_entries"] > 0
                    for pair in realized["pairwise"].values()
                )
            )


if __name__ == "__main__":
    unittest.main()
