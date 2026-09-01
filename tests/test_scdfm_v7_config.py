from __future__ import annotations

import tomllib
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs/scdfm/vcc2026_v7_gamma1.toml"


def _all_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        keys = set(value)
        for nested in value.values():
            keys.update(_all_keys(nested))
        return keys
    if isinstance(value, list):
        keys: set[str] = set()
        for nested in value:
            keys.update(_all_keys(nested))
        return keys
    return set()


class ScdfmV7ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CONFIG_PATH.open("rb") as handle:
            cls.config = tomllib.load(handle)

    def test_state_anchor_is_frozen_at_gamma_one(self) -> None:
        anchor = self.config["anchor"]
        self.assertEqual(anchor["selected_arm"], "g100")
        self.assertEqual(anchor["state_effect_weight"], 1.0)
        self.assertTrue(anchor["immutable_during_scdfm_search"])
        self.assertEqual(
            anchor["selection_receipt_sha256"],
            "cd6618f38f0b93c8e7b3c63a8a2c1a31b87360ead5cc91b3f7e43898c100d312",
        )

    def test_scdfm_loss_does_not_reuse_the_gamma_name(self) -> None:
        self.assertNotIn("gamma", _all_keys(self.config))
        self.assertEqual(self.config["model"]["mmd_weight"], 0.5)

    def test_registered_residual_factor_screen_includes_exact_anchor(self) -> None:
        self.assertEqual(
            self.config["search"]["scdfm_residual_weights"],
            [0.0, 0.05, 0.10, 0.20],
        )
        self.assertTrue(
            self.config["promotion"][
                "require_centroid_identity_for_distribution_only_arm"
            ]
        )

    def test_challenge_contract_and_submission_firewall_are_explicit(self) -> None:
        challenge = self.config["challenge"]
        self.assertEqual(challenge["target_count"], 300)
        self.assertEqual(challenge["gene_count"], 18_533)
        self.assertEqual(challenge["cells_per_context_target"], 400)
        self.assertFalse(challenge["measured_challenge_perturbation_data_allowed"])
        self.assertFalse(self.config["experiment"]["official_submission_allowed"])

    def test_sealed_hepg2_data_cannot_train_the_count_emitter(self) -> None:
        datasets = {item["name"]: item for item in self.config["datasets"]}
        self.assertEqual(
            datasets["nadig_hepg2"]["role"],
            "sealed_primary_zero_shot_validation",
        )
        self.assertFalse(datasets["nadig_hepg2"]["allowed_for_count_emitter"])
        self.assertFalse(datasets["scdfm_norman"]["allowed_for_count_emitter"])
        self.assertEqual(
            datasets["scdfm_norman"]["role"],
            "upstream_reproduction_only_crispr_activation",
        )

    def test_jurkat_non_harm_partition_is_sealed_from_fitting(self) -> None:
        search = self.config["search"]
        self.assertEqual(search["secondary_non_harm_context"], "jurkat")
        self.assertTrue(search["secondary_non_harm_excluded_from_fit"])
        self.assertTrue(search["secondary_non_harm_manifest_required_before_training"])
        self.assertEqual(
            search["secondary_non_harm_manifest"],
            "artifacts/scdfm/v7/splits/jurkat_non_harm.json",
        )
        datasets = {item["name"]: item for item in self.config["datasets"]}
        self.assertEqual(
            datasets["nadig_jurkat"]["role"],
            "distribution_training_excluding_sealed_non_harm_partition",
        )

    def test_esm2_provenance_is_fail_closed_until_exact_assets_are_bound(self) -> None:
        provenance = self.config["target_conditioning_provenance"]
        required_unresolved_fields = (
            "model_id",
            "model_revision",
            "weights_sha256",
            "protein_sequence_source",
            "gene_to_protein_mapping",
            "gene_to_protein_mapping_sha256",
            "pooling_rule",
            "embedding_receipt",
            "embedding_receipt_sha256",
        )
        self.assertEqual(provenance["status"], "UNRESOLVED_REQUIRED")
        self.assertEqual(provenance["embedding_dimension"], 5120)
        for field in required_unresolved_fields:
            self.assertEqual(provenance[field], "UNRESOLVED_REQUIRED")
        self.assertFalse(provenance["provenance_complete"])
        self.assertFalse(provenance["gate_1b_allowed"])


if __name__ == "__main__":
    unittest.main()
