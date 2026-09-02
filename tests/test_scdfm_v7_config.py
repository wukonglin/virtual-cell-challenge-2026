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
        self.assertFalse(self.config["experiment"]["training_ready"])
        self.assertEqual(
            self.config["experiment"]["training_blocker"],
            "NO_PORTABLE_TRAINER_CONSUMING_HASH_PINNED_GATE",
        )
        self.assertEqual(
            self.config["model"]["target_conditioning"],
            "arc_official_opaque_gene_features_5120_projected",
        )

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

        registration = search["jurkat_non_harm_registration"]
        self.assertEqual(
            registration["schema"],
            "vcc-scdfm-jurkat-non-harm-registration-v1",
        )
        self.assertEqual(registration["raw_shape"], [262_956, 8_882])
        self.assertEqual(registration["raw_treated_target_count"], 2_393)
        self.assertEqual(registration["raw_control_cells"], 12_013)
        self.assertEqual(registration["desired_held_targets"], 300)
        self.assertEqual(registration["minimum_treated_cells"], 64)
        self.assertEqual(registration["cluster_count"], 30)
        self.assertEqual(registration["kmeans_n_init"], 20)
        self.assertEqual(registration["kmeans_init"], "k-means++")
        self.assertEqual(registration["kmeans_max_iter"], 300)
        self.assertEqual(registration["kmeans_tol"], 0.0001)
        self.assertEqual(registration["kmeans_algorithm"], "lloyd")
        self.assertEqual(registration["thread_policy"], "threadpoolctl-limit-1")
        self.assertEqual(registration["construction_environment"], ".venv-state")
        self.assertEqual(
            registration["construction_environment_role"],
            "offline_split_audit_only",
        )
        self.assertFalse(registration["trainer_may_reconstruct_clusters"])
        self.assertFalse(registration["trainer_may_refit_or_relabel_clusters"])
        self.assertTrue(registration["training_preflight_required"])
        self.assertEqual(
            registration["training_preflight_schema"],
            "vcc-scdfm-jurkat-training-preflight-v1",
        )
        self.assertEqual(
            registration["training_preflight_receipt"],
            "artifacts/scdfm/v7/training/jurkat_training_preflight.json",
        )
        self.assertEqual(registration["expected_state_support_target_union"], 197)
        self.assertEqual(
            registration["expected_state_support_target_union_sha256"],
            "53ebab565d7e5c9cc12c431e3207ab20416e247bceffe4bca7a4d90f63911ace",
        )
        self.assertEqual(len(registration["state_support_sources"]), 6)
        for source in registration["state_support_sources"]:
            self.assertEqual(
                set(source),
                {"path", "size_bytes", "sha256", "target_count", "target_list_sha256"},
            )
            self.assertGreater(source["size_bytes"], 0)
            self.assertEqual(len(source["sha256"]), 64)
            self.assertGreater(source["target_count"], 0)
            self.assertEqual(len(source["target_list_sha256"]), 64)
        self.assertEqual(
            registration["global_exclusion_scopes"],
            [
                "model_fit",
                "checkpoint_selection",
                "normalization_fit",
                "feature_selection",
                "graph_construction",
                "decoder_fit",
                "hyperparameter_selection",
            ],
        )

    def test_esm2_provenance_is_fail_closed_until_exact_assets_are_bound(self) -> None:
        provenance = self.config["target_conditioning_provenance"]
        required_unresolved_fields = (
            "model_id",
            "model_revision",
            "weights_artifact",
            "weights_sha256",
            "protein_sequence_source",
            "protein_sequence_source_sha256",
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
        self.assertFalse(provenance["feature_derivation_replay_implemented"])
        self.assertFalse(
            provenance["full_feature_map_recomputed_from_registered_upstream_assets"]
        )

    def test_portable_trainer_contract_is_explicitly_not_ready(self) -> None:
        portable = self.config["portable_training"]
        self.assertEqual(
            portable["schema"],
            "vcc-scdfm-portable-training-contract-v1",
        )
        self.assertFalse(portable["training_ready"])
        self.assertFalse(portable["trainer_entrypoint_implemented"])
        self.assertTrue(portable["offline_split_audit_only"])
        self.assertTrue(portable["offline_audit_may_reconstruct_kmeans"])
        self.assertTrue(portable["portable_trainer_must_not_import_or_run_kmeans"])
        self.assertTrue(
            portable["portable_trainer_must_consume_hash_pinned_manifest"]
        )
        self.assertTrue(
            portable["portable_trainer_must_consume_hash_pinned_preflight"]
        )
        self.assertEqual(
            portable["portable_hash_lock_status"],
            "UNRESOLVED_REQUIRED",
        )

    def test_opaque_esm2_artifact_authentication_does_not_open_gate_1b(self) -> None:
        artifact = self.config["target_conditioning_artifact_authentication"]
        self.assertEqual(artifact["status"], "authenticated_opaque_artifact")
        self.assertEqual(
            artifact["raw_receipt_schema"],
            "vcc-arc-esm2-target-features-authentication-gate1c-v2",
        )
        self.assertEqual(artifact["raw_receipt_size_bytes"], 4_864)
        self.assertEqual(
            artifact["raw_receipt_sha256"],
            "a858816ce4ac4dc6581d0dda698c9b146bb6e33c70f132078317c3fd0b99be62",
        )
        self.assertEqual(
            artifact["raw_receipt"],
            "dataset/state_support/receipts/esm2_target_features_gate1c_v2.json",
        )
        self.assertEqual(
            artifact["source_object_generation"],
            "1763329752965344",
        )
        self.assertEqual(artifact["source_object_crc32c_base64"], "MBP7uA==")
        self.assertFalse(artifact["source_url_claimed_permanently_immutable"])
        self.assertTrue(artifact["official_opaque_artifact_training_allowed"])
        self.assertTrue(artifact["gate_1c_allowed"])
        self.assertEqual(artifact["feature_count"], 19_790)
        self.assertEqual(artifact["embedding_dimension"], 5_120)
        self.assertEqual(artifact["vcc_target_coverage"], 300)
        self.assertEqual(
            artifact["artifact_sha256"],
            "a210e1cc7901513999b2bca3836ba9e2f203cd008be4e9a9d6412a2267de9748",
        )
        self.assertEqual(
            artifact["canonical_map_sha256"],
            "e41e97901c2e208e27c631f67c533f01bab2314ef1f9367190f832256866b399",
        )
        self.assertEqual(
            artifact["ordered_target_feature_subset_sha256"],
            "2149553a0783bb3959403e62913b1ff9983e2abc2a1e99025608c0660d12d2d8",
        )
        self.assertFalse(artifact["upstream_model_metadata_provided"])
        self.assertTrue(
            artifact["does_not_satisfy_target_conditioning_provenance"]
        )
        self.assertFalse(artifact["provenance_complete"])
        self.assertFalse(artifact["gate_1b_allowed"])


if __name__ == "__main__":
    unittest.main()
