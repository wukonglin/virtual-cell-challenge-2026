"""Adversarial tests for the V7 Jurkat non-harm split firewall."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
import sklearn
import torch


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPTS = REPOSITORY / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from scdfm_jurkat_non_harm import (  # noqa: E402
    OFFICIAL_GATE_1C,
    OPAQUE_ESM2_RECEIPT_SCHEMA,
    JurkatManifestError,
    _TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    assert_loaded_training_batch_is_clean,
    authenticate_manifest as _authenticate_manifest,
    authenticate_training_preflight as _authenticate_training_preflight,
    canonical_feature_key_sha256,
    canonical_feature_map_sha256,
    construct_manifest as _construct_manifest,
    construct_training_preflight as _construct_training_preflight,
    load_training_preflight_gate as _load_training_preflight_gate,
    read_state_support_targets,
    validate_official_gate_1c_receipt,
    write_json_atomic,
)


def construct_manifest(config: Path) -> dict[str, object]:
    return _construct_manifest(
        config,
        _test_only_lineage_bypass=_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    )


def authenticate_manifest(config: Path, manifest: Path) -> dict[str, object]:
    return _authenticate_manifest(
        config,
        manifest,
        _test_only_lineage_bypass=_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    )


def construct_training_preflight(config: Path, manifest: Path) -> dict[str, object]:
    return _construct_training_preflight(
        config,
        manifest,
        _test_only_lineage_bypass=_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    )


def authenticate_training_preflight(
    config: Path,
    manifest: Path,
    preflight: Path,
) -> dict[str, object]:
    return _authenticate_training_preflight(
        config,
        manifest,
        preflight,
        _test_only_lineage_bypass=_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    )


def load_training_preflight_gate(config: Path, manifest: Path, preflight: Path):
    return _load_training_preflight_gate(
        config,
        manifest,
        preflight,
        _test_only_lineage_bypass=_TEST_ONLY_SYNTHETIC_LINEAGE_BYPASS,
    )


def official_gate_1c_registration() -> dict[str, object]:
    source_object = OFFICIAL_GATE_1C["source_object"]
    return {
        "status": "authenticated_opaque_artifact",
        "role": "artifact_identity_and_structure_only",
        "official_opaque_artifact": True,
        "opaque_artifact_training_mode": "official_opaque_artifact",
        "official_opaque_artifact_training_allowed": True,
        "gate_1c_allowed": True,
        "gate_1c_scope": (
            "authenticated_arc_supplied_artifact_identity_and_structure_only"
        ),
        "gate_1b_allowed": False,
        "provenance_complete": False,
        "does_not_satisfy_target_conditioning_provenance": True,
        "vcc_target_coverage": OFFICIAL_GATE_1C["target_count"],
        "upstream_model_metadata_provided": False,
        "raw_receipt": OFFICIAL_GATE_1C["receipt_path"],
        "raw_receipt_schema": OPAQUE_ESM2_RECEIPT_SCHEMA,
        "raw_receipt_size_bytes": OFFICIAL_GATE_1C["receipt_size_bytes"],
        "raw_receipt_sha256": OFFICIAL_GATE_1C["receipt_sha256"],
        "source_url": OFFICIAL_GATE_1C["source_url"],
        "source_object_provider": source_object["provider"],
        "source_object_generation": source_object["generation"],
        "source_object_last_modified": source_object["last_modified"],
        "source_object_size_bytes": source_object["size_bytes"],
        "source_object_crc32c_base64": source_object["crc32c_base64"],
        "source_object_crc32c_hex": source_object["crc32c_hex"],
        "source_object_component_count": source_object["component_count"],
        "source_object_metadata_scope": source_object["metadata_scope"],
        "source_url_claimed_permanently_immutable": source_object[
            "source_url_claimed_permanently_immutable"
        ],
        "source_archive": f"dataset/{OFFICIAL_GATE_1C['archive_filename']}",
        "source_archive_size_bytes": OFFICIAL_GATE_1C["archive_size_bytes"],
        "source_archive_sha256": OFFICIAL_GATE_1C["archive_sha256"],
        "archive_member": OFFICIAL_GATE_1C["member_name"],
        "archive_member_size_bytes": OFFICIAL_GATE_1C["member_size_bytes"],
        "archive_member_sha256": OFFICIAL_GATE_1C["member_sha256"],
        "archive_member_crc32": OFFICIAL_GATE_1C["member_crc32"],
        "artifact": f"dataset/{OFFICIAL_GATE_1C['feature_filename']}",
        "artifact_size_bytes": OFFICIAL_GATE_1C["member_size_bytes"],
        "artifact_sha256": OFFICIAL_GATE_1C["member_sha256"],
        "feature_count": OFFICIAL_GATE_1C["feature_count"],
        "embedding_dimension": OFFICIAL_GATE_1C["embedding_dimension"],
        "tensor_dtype": OFFICIAL_GATE_1C["tensor_dtype"],
        "tensor_layout": OFFICIAL_GATE_1C["tensor_layout"],
        "canonical_key_list_sha256": OFFICIAL_GATE_1C[
            "canonical_key_list_sha256"
        ],
        "canonical_map_sha256": OFFICIAL_GATE_1C["canonical_map_sha256"],
        "target_manifest_size_bytes": OFFICIAL_GATE_1C[
            "target_manifest_size_bytes"
        ],
        "target_manifest_sha256": OFFICIAL_GATE_1C["target_manifest_sha256"],
        "target_count": OFFICIAL_GATE_1C["target_count"],
        "ordered_target_list_sha256": OFFICIAL_GATE_1C[
            "ordered_target_list_sha256"
        ],
        "ordered_target_feature_subset_sha256": OFFICIAL_GATE_1C[
            "ordered_target_feature_subset_sha256"
        ],
    }


def official_gate_1c_receipt() -> dict[str, object]:
    return {
        "schema": OPAQUE_ESM2_RECEIPT_SCHEMA,
        "status": "passed",
        "contract": {
            "artifact_role": "opaque_arc_supplied_target_features",
            "authentication_scope": "artifact_identity_and_structure_only",
            "canonical_tensor_byte_order": "little-endian-float32",
            "does_not_establish_upstream_model_provenance": True,
            "provenance_complete": False,
            "gate_1b_allowed": False,
            "opaque_artifact_training_mode": "official_opaque_artifact",
            "official_opaque_artifact_training_allowed": True,
            "gate_1c_allowed": True,
            "gate_1c_scope": (
                "authenticated_arc_supplied_artifact_identity_and_structure_only"
            ),
        },
        "archive": {
            "source_url": OFFICIAL_GATE_1C["source_url"],
            "source_object": dict(OFFICIAL_GATE_1C["source_object"]),
            "local_filename": OFFICIAL_GATE_1C["archive_filename"],
            "size_bytes": OFFICIAL_GATE_1C["archive_size_bytes"],
            "sha256": OFFICIAL_GATE_1C["archive_sha256"],
            "member": {
                "name": OFFICIAL_GATE_1C["member_name"],
                "size_bytes": OFFICIAL_GATE_1C["member_size_bytes"],
                "sha256": OFFICIAL_GATE_1C["member_sha256"],
                "zip_crc32": OFFICIAL_GATE_1C["member_crc32"],
                "encrypted": False,
                "symbolic_link": False,
            },
        },
        "feature_artifact": {
            "local_filename": OFFICIAL_GATE_1C["feature_filename"],
            "size_bytes": OFFICIAL_GATE_1C["member_size_bytes"],
            "sha256": OFFICIAL_GATE_1C["member_sha256"],
            "expected_sha256": OFFICIAL_GATE_1C["member_sha256"],
            "post_load_sha256": OFFICIAL_GATE_1C["member_sha256"],
            "matches_registered_archive_member": True,
            "post_load_sha256_matches_preload": True,
            "restricted_weights_only_load": True,
            "sha256_matches_expected_when_provided": True,
            "map_location": "cpu",
            "explicit_safe_globals": [],
            "structure": {
                "feature_count": OFFICIAL_GATE_1C["feature_count"],
                "embedding_dimension": OFFICIAL_GATE_1C["embedding_dimension"],
                "tensor_dtype": OFFICIAL_GATE_1C["tensor_dtype"],
                "tensor_layout": OFFICIAL_GATE_1C["tensor_layout"],
                "tensor_device": "cpu",
                "canonical_key_list_sha256": OFFICIAL_GATE_1C[
                    "canonical_key_list_sha256"
                ],
                "canonical_map_sha256": OFFICIAL_GATE_1C[
                    "canonical_map_sha256"
                ],
                "all_values_finite": True,
                "all_vectors_nonzero": True,
            },
        },
        "target_manifest": {
            "local_filename": OFFICIAL_GATE_1C["target_manifest_filename"],
            "size_bytes": OFFICIAL_GATE_1C["target_manifest_size_bytes"],
            "sha256": OFFICIAL_GATE_1C["target_manifest_sha256"],
            "target_count": OFFICIAL_GATE_1C["target_count"],
            "target_column": "target_gene",
            "targets_are_unique": True,
            "ordered_target_list_sha256": OFFICIAL_GATE_1C[
                "ordered_target_list_sha256"
            ],
        },
        "vcc_target_coverage": {
            "covered_targets": OFFICIAL_GATE_1C["target_count"],
            "requested_targets": OFFICIAL_GATE_1C["target_count"],
            "missing_targets": 0,
            "ordered_target_feature_subset_sha256": OFFICIAL_GATE_1C[
                "ordered_target_feature_subset_sha256"
            ],
        },
        "upstream_provenance": {
            "model_id": "UNKNOWN_NOT_PROVIDED",
            "model_revision": "UNKNOWN_NOT_PROVIDED",
            "weights_sha256": "UNKNOWN_NOT_PROVIDED",
            "protein_sequence_source": "UNKNOWN_NOT_PROVIDED",
            "gene_to_protein_mapping": "UNKNOWN_NOT_PROVIDED",
            "pooling_rule": "UNKNOWN_NOT_PROVIDED",
            "inferred_from_embedding_dimension": False,
        },
        "checks": {"all_registered_checks_passed": True},
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def write_string_dataset(group: h5py.Group, name: str, values: list[str]) -> None:
    group.create_dataset(name, data=np.asarray(values, dtype=h5py.string_dtype("utf-8")))


class SyntheticProject:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = root / "configs/scdfm/v7.toml"
        self.raw = root / "dataset/raw/jurkat.h5ad"
        self.features = root / "dataset/features.pt"
        self.receipt = root / "dataset/features_receipt.json"
        self.historical = root / "dataset/historical.csv"
        self.support = root / "dataset/support.h5"
        self.weights = root / "dataset/esm2_weights.bin"
        self.sequences = root / "dataset/proteins.fasta"
        self.mapping = root / "dataset/gene_to_protein.tsv"
        self.output = root / "artifacts/scdfm/v7/splits/jurkat_non_harm.json"
        self.preflight = root / "artifacts/scdfm/v7/training/preflight.json"
        for path in (
            self.config,
            self.raw,
            self.features,
            self.receipt,
            self.historical,
            self.support,
            self.weights,
            self.sequences,
            self.mapping,
            self.output,
            self.preflight,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)
        self._write_raw()
        self._write_support()
        self.historical.write_text("target_gene\nold\n", encoding="utf-8")
        self.weights.write_bytes(b"authenticated synthetic model weights\n")
        self.sequences.write_text(">P_A\nMAAAA\n>P_B\nMBBBB\n", encoding="utf-8")
        self.mapping.write_text(
            "gene\tprotein\na\tP_A\nb\tP_B\nc\tP_A\nd\tP_B\n",
            encoding="utf-8",
        )
        self.feature_map = {
            "a": torch.tensor([1.0, 0.01, 0.0, 0.0], dtype=torch.float32),
            "b": torch.tensor([1.0, 0.02, 0.0, 0.0], dtype=torch.float32),
            "c": torch.tensor([0.01, 1.0, 0.0, 0.0], dtype=torch.float32),
            "d": torch.tensor([0.02, 1.0, 0.0, 0.0], dtype=torch.float32),
            "old": torch.tensor([1.0, 0.03, 0.0, 0.0], dtype=torch.float32),
            "support": torch.tensor([0.03, 1.0, 0.0, 0.0], dtype=torch.float32),
        }
        torch.save(self.feature_map, self.features)
        self._write_receipt(provenance_complete=True)
        self._write_config()

    def _write_raw(self) -> None:
        labels = ["non-targeting", "non-targeting"]
        for target in ("a", "b", "c", "d", "old", "support"):
            labels.extend([target, target, target])
        cells = [f"cell-{index:02d}" for index in range(len(labels))]
        with h5py.File(self.raw, "w") as data:
            # Deliberately non-finite expression proves split construction does
            # not interpret treated expression values.
            data.create_dataset("X", data=np.full((len(labels), 8), np.nan, dtype=np.float32))
            obs = data.create_group("obs")
            write_string_dataset(obs, "cell_barcode", cells)
            write_string_dataset(obs, "gene", labels)
            write_string_dataset(obs, "gem_group", ["batch-1"] * len(labels))
            write_string_dataset(obs, "sgID_AB", [f"guide-{label}" for label in labels])
            var = data.create_group("var")
            write_string_dataset(
                var,
                "gene_name",
                ["a", "b", "c", "d", "old", "support", "g1", "g2"],
            )

    def _write_support(self) -> None:
        with h5py.File(self.support, "w") as data:
            data.create_dataset("X", data=np.full((2, 1), 99.0, dtype=np.float32))
            obs = data.create_group("obs")
            write_string_dataset(obs, "target_gene", ["support", "non-targeting"])

    def _write_receipt(self, *, provenance_complete: bool) -> None:
        receipt = {
            "schema": "vcc-esm2-target-conditioning-provenance-v1",
            "status": "passed",
            "contract": {
                "provenance_complete": provenance_complete,
                "gate_1b_allowed": provenance_complete,
            },
            "feature_artifact": {
                "local_filename": self.features.name,
                "size_bytes": self.features.stat().st_size,
                "sha256": sha256_file(self.features),
                "structure": {
                    "feature_count": len(self.feature_map),
                    "embedding_dimension": 4,
                    "canonical_key_list_sha256": canonical_feature_key_sha256(
                        list(self.feature_map)
                    ),
                    "canonical_map_sha256": canonical_feature_map_sha256(self.feature_map),
                },
            },
            "upstream_provenance": {
                "model_id": "synthetic-esm2",
                "model_revision": "test-revision",
                "pooling_rule": "mean-residue",
                "weights_artifact": {
                    "filename": self.weights.name,
                    "size_bytes": self.weights.stat().st_size,
                    "sha256": sha256_file(self.weights),
                },
                "protein_sequence_artifact": {
                    "filename": self.sequences.name,
                    "size_bytes": self.sequences.stat().st_size,
                    "sha256": sha256_file(self.sequences),
                },
                "gene_to_protein_mapping_artifact": {
                    "filename": self.mapping.name,
                    "size_bytes": self.mapping.stat().st_size,
                    "sha256": sha256_file(self.mapping),
                },
            },
        }
        self.receipt.write_text(json.dumps(receipt), encoding="utf-8")

    def _relative(self, path: Path) -> str:
        return path.relative_to(self.root).as_posix()

    def _write_config(self) -> None:
        text = f'''[experiment]
schema = "vcc-public-v7-scdfm-experiment-v1"

[target_conditioning_provenance]
embedding_receipt = "{self._relative(self.receipt)}"
embedding_receipt_sha256 = "{sha256_file(self.receipt)}"
model_id = "synthetic-esm2"
model_revision = "test-revision"
weights_artifact = "{self._relative(self.weights)}"
weights_sha256 = "{sha256_file(self.weights)}"
protein_sequence_source = "{self._relative(self.sequences)}"
protein_sequence_source_sha256 = "{sha256_file(self.sequences)}"
gene_to_protein_mapping = "{self._relative(self.mapping)}"
gene_to_protein_mapping_sha256 = "{sha256_file(self.mapping)}"
pooling_rule = "mean-residue"
embedding_dimension = 4
provenance_complete = true
gate_1b_allowed = true

[search]
secondary_non_harm_manifest = "{self._relative(self.output)}"

[search.jurkat_non_harm_registration]
schema = "vcc-scdfm-jurkat-non-harm-registration-v1"
raw_jurkat = "{self._relative(self.raw)}"
raw_size_bytes = {self.raw.stat().st_size}
raw_sha256 = "{sha256_file(self.raw)}"
raw_shape = [20, 8]
raw_treated_target_count = 6
raw_control_cells = 2
esm2_artifact = "{self._relative(self.features)}"
historical_panel = "{self._relative(self.historical)}"
historical_panel_sha256 = "{sha256_file(self.historical)}"
state_support_sources = [
  {{ path = "{self._relative(self.support)}", size_bytes = {self.support.stat().st_size}, sha256 = "{sha256_file(self.support)}", target_count = 1, target_list_sha256 = "{hashlib.sha256(json.dumps(['support'], ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()}" }},
]
expected_state_support_target_union = 1
expected_state_support_target_union_sha256 = "{hashlib.sha256(json.dumps(['support'], ensure_ascii=False, separators=(',', ':'), sort_keys=True).encode()).hexdigest()}"
seed = 20260901
minimum_treated_cells = 2
desired_held_targets = 2
cluster_count = 2
kmeans_n_init = 20
kmeans_init = "k-means++"
kmeans_max_iter = 300
kmeans_tol = 0.0001
kmeans_algorithm = "lloyd"
thread_policy = "threadpoolctl-limit-1"
numpy_version = "{np.__version__}"
scikit_learn_version = "{sklearn.__version__}"
training_preflight_required = true
training_preflight_schema = "vcc-scdfm-jurkat-training-preflight-v1"
training_preflight_receipt = "artifacts/scdfm/v7/training/preflight.json"
global_exclusion_scopes = [
  "model_fit",
  "checkpoint_selection",
  "normalization_fit",
  "feature_selection",
  "graph_construction",
  "decoder_fit",
  "hyperparameter_selection",
]
'''
        self.config.write_text(text, encoding="utf-8")


class JurkatNonHarmManifestTests(unittest.TestCase):
    def test_gate_1c_accepts_only_exact_official_opaque_artifact_identity(self) -> None:
        registration = official_gate_1c_registration()
        receipt = official_gate_1c_receipt()
        artifact = validate_official_gate_1c_receipt(receipt, registration)
        self.assertEqual(artifact["sha256"], OFFICIAL_GATE_1C["member_sha256"])

        registration_mutations = (
            ("legacy receipt schema", "raw_receipt_schema", "vcc-arc-esm2-target-features-authentication-v1"),
            ("receipt size", "raw_receipt_size_bytes", 4_863),
            ("receipt hash", "raw_receipt_sha256", "0" * 64),
            (
                "official training mode",
                "official_opaque_artifact_training_allowed",
                False,
            ),
            ("source provider", "source_object_provider", "self-asserted"),
            ("archive member hash", "archive_member_sha256", "0" * 64),
        )
        for label, key, value in registration_mutations:
            with self.subTest(label=label):
                changed_registration = dict(registration)
                changed_registration[key] = value
                with self.assertRaises(JurkatManifestError):
                    validate_official_gate_1c_receipt(
                        receipt,
                        changed_registration,
                    )

        mutations = (
            ("generation", lambda value: value["archive"]["source_object"].update({"generation": "0"})),
            ("canonical map", lambda value: value["feature_artifact"]["structure"].update({"canonical_map_sha256": "0" * 64})),
            ("Gate 1b", lambda value: value["contract"].update({"gate_1b_allowed": True})),
            ("target order", lambda value: value["target_manifest"].update({"ordered_target_list_sha256": "0" * 64})),
        )
        for label, mutate in mutations:
            with self.subTest(label=label):
                changed = json.loads(json.dumps(receipt))
                mutate(changed)
                with self.assertRaises(JurkatManifestError):
                    validate_official_gate_1c_receipt(changed, registration)

        self_asserted_complete = json.loads(json.dumps(receipt))
        self_asserted_complete["schema"] = "vcc-esm2-target-conditioning-provenance-v1"
        with self.assertRaisesRegex(JurkatManifestError, "Bad Gate 1c receipt schema"):
            validate_official_gate_1c_receipt(
                self_asserted_complete,
                registration,
            )
        legacy_v1 = json.loads(json.dumps(receipt))
        legacy_v1["schema"] = "vcc-arc-esm2-target-features-authentication-v1"
        with self.assertRaisesRegex(JurkatManifestError, "Bad Gate 1c receipt schema"):
            validate_official_gate_1c_receipt(legacy_v1, registration)

    def test_production_gate_rejects_self_asserted_complete_lineage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            with self.assertRaisesRegex(
                JurkatManifestError,
                "No authenticated official opaque ESM2 artifact is registered",
            ):
                _construct_manifest(project.config)
            self.assertFalse(project.output.exists())

    def test_metadata_only_whole_cluster_manifest_and_authenticator(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            manifest = construct_manifest(project.config)
            self.assertEqual(manifest["status"], "sealed_test_only")
            self.assertFalse(
                manifest["lineage_firewall"]["production_training_ready"]
            )
            self.assertFalse(manifest["data_firewall"]["expression_matrix_values_accessed"])
            scored = {
                row["target_gene"] for row in manifest["partition"]["scored_held_targets"]
            }
            self.assertEqual(len(scored), 2)
            self.assertFalse(scored & {"old", "support"})
            self.assertTrue(
                scored.issubset(set(manifest["partition"]["global_exclusion_targets"]))
            )
            self.assertTrue(
                manifest["assertions"][
                    "no_held_target_or_cluster_member_may_enter_any_fit_transform_or_graph"
                ]
            )
            self.assertEqual(
                manifest["partition"]["cluster_assignment_sha256"],
                manifest["partition"]["repeated_cluster_assignment_sha256"],
            )
            self.assertEqual(
                len(manifest["partition"]["cluster_assignments"]),
                manifest["partition"]["clustering_universe_target_count"],
            )
            self.assertEqual(
                set(manifest["inputs"]["target_conditioning_upstream_artifacts"]),
                {"weights", "protein_sequences", "gene_to_protein_mapping"},
            )
            write_json_atomic(project.output, manifest)
            receipt = authenticate_manifest(project.config, project.output)
            self.assertEqual(receipt["status"], "passed_test_only")

            preflight = construct_training_preflight(project.config, project.output)
            self.assertEqual(preflight["status"], "passed_test_only")
            self.assertEqual(
                preflight["sealed_manifest"]["sha256"], sha256_file(project.output)
            )
            self.assertTrue(
                all(
                    dataset["held_targets_absent_from_allowed_rows"]
                    for dataset in preflight["datasets"]
                )
            )
            write_json_atomic(project.preflight, preflight)
            with self.assertRaisesRegex(
                JurkatManifestError,
                "Portable trainer integration is not implemented",
            ):
                _load_training_preflight_gate(
                    project.config,
                    project.output,
                    project.preflight,
                )
            preflight_authentication = authenticate_training_preflight(
                project.config,
                project.output,
                project.preflight,
            )
            self.assertEqual(
                preflight_authentication["status"],
                "passed_test_only",
            )
            gate = load_training_preflight_gate(
                project.config,
                project.output,
                project.preflight,
            )
            self.assertFalse(gate.production_training_ready)
            with h5py.File(project.raw, "r") as data:
                raw_labels = [
                    value.decode("utf-8") if isinstance(value, bytes) else str(value)
                    for value in data["obs/gene"][:]
                ]
            allowed = gate.allowed_source_row_indices["nadig_jurkat"]
            gate.authorize_complete_selection(
                "nadig_jurkat",
                allowed,
                [raw_labels[index] for index in allowed],
                "model_fit",
            )
            with self.assertRaisesRegex(JurkatManifestError, "differs from preflight"):
                gate.authorize_complete_selection(
                    "nadig_jurkat",
                    allowed[:-1],
                    [raw_labels[index] for index in allowed[:-1]],
                    "normalization_fit",
                )
            held = set(manifest["partition"]["global_exclusion_targets"])
            with self.assertRaisesRegex(JurkatManifestError, "contains sealed target"):
                assert_loaded_training_batch_is_clean(
                    [next(iter(held)), "non-targeting"], held, "synthetic batch"
                )
            with self.assertRaisesRegex(JurkatManifestError, "contains sealed target"):
                gate.assert_batch([next(iter(held))], "graph_construction")

    def test_incomplete_esm2_provenance_fails_before_raw_input_is_opened(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            project._write_receipt(provenance_complete=False)
            project._write_config()
            project.raw.unlink()
            with self.assertRaisesRegex(JurkatManifestError, "upstream provenance is incomplete"):
                construct_manifest(project.config)
            self.assertFalse(project.output.exists())

    def test_opaque_feature_receipt_cannot_be_upgraded_by_flipping_booleans(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            receipt = json.loads(project.receipt.read_text(encoding="utf-8"))
            receipt["schema"] = "vcc-arc-esm2-target-features-authentication-v1"
            receipt["contract"]["provenance_complete"] = True
            receipt["contract"]["gate_1b_allowed"] = True
            project.receipt.write_text(json.dumps(receipt), encoding="utf-8")
            project._write_config()
            with self.assertRaisesRegex(
                JurkatManifestError,
                "opaque feature receipt cannot open Gate 1b",
            ):
                construct_manifest(project.config)

    def test_symlinked_receipt_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            real_receipt = project.receipt.with_name("real.json")
            project.receipt.replace(real_receipt)
            project.receipt.symlink_to(real_receipt.name)
            project._write_config()
            with self.assertRaisesRegex(JurkatManifestError, "Unable to open ESM2 provenance receipt"):
                construct_manifest(project.config)

    def test_duplicate_receipt_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            project.receipt.write_text(
                '{"schema":"vcc-esm2-target-conditioning-provenance-v1",'
                '"status":"passed","status":"passed"}',
                encoding="utf-8",
            )
            project._write_config()
            with self.assertRaisesRegex(JurkatManifestError, "Duplicate JSON key"):
                construct_manifest(project.config)

    def test_tampered_embedding_artifact_is_rejected_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            with project.features.open("ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(JurkatManifestError, "ESM2 artifact size changed"):
                construct_manifest(project.config)
            self.assertFalse(project.output.exists())

    def test_self_asserted_upstream_hashes_cannot_authenticate_actual_files(self) -> None:
        cases = (
            ("weights_artifact", "weights_sha256", "weights"),
            (
                "protein_sequence_artifact",
                "protein_sequence_source_sha256",
                "protein_sequences",
            ),
            (
                "gene_to_protein_mapping_artifact",
                "gene_to_protein_mapping_sha256",
                "gene_to_protein_mapping",
            ),
        )
        for receipt_field, config_field, error_label in cases:
            with self.subTest(receipt_field=receipt_field):
                with tempfile.TemporaryDirectory() as temporary:
                    project = SyntheticProject(Path(temporary))
                    receipt = json.loads(project.receipt.read_text(encoding="utf-8"))
                    actual = receipt["upstream_provenance"][receipt_field]["sha256"]
                    fake = "f" * 64
                    receipt["upstream_provenance"][receipt_field]["sha256"] = fake
                    project.receipt.write_text(json.dumps(receipt), encoding="utf-8")
                    project._write_config()
                    config_text = project.config.read_text(encoding="utf-8")
                    project.config.write_text(
                        config_text.replace(
                            f'{config_field} = "{actual}"',
                            f'{config_field} = "{fake}"',
                        ),
                        encoding="utf-8",
                    )
                    with self.assertRaisesRegex(
                        JurkatManifestError,
                        f"{error_label} SHA-256 changed",
                    ):
                        construct_manifest(project.config)

    def test_same_count_state_support_substitution_is_rejected_by_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            original_size = project.support.stat().st_size
            with h5py.File(project.support, "r+") as data:
                data["obs/target_gene"][0] = "intrude"
            self.assertEqual(project.support.stat().st_size, original_size)
            with self.assertRaisesRegex(JurkatManifestError, "STATE support SHA-256 changed"):
                construct_manifest(project.config)

    def test_same_count_state_target_substitution_fails_exact_target_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            with h5py.File(project.support, "r+") as data:
                data["obs/target_gene"][0] = "intrude"
            # Re-registering the changed container hash cannot bypass the
            # independently frozen target-list identity.
            project._write_config()
            with self.assertRaisesRegex(JurkatManifestError, "target-list SHA-256 changed"):
                construct_manifest(project.config)

    def test_state_support_union_hash_is_independently_bound(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            config_text = project.config.read_text(encoding="utf-8")
            expected = hashlib.sha256(
                json.dumps(
                    ["support"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            project.config.write_text(
                config_text.replace(
                    f'expected_state_support_target_union_sha256 = "{expected}"',
                    f'expected_state_support_target_union_sha256 = "{"0" * 64}"',
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(JurkatManifestError, "target-union SHA-256 drift"):
                construct_manifest(project.config)

    def test_atomic_writer_refuses_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            write_json_atomic(path, {"first": True})
            with self.assertRaisesRegex(FileExistsError, "Refusing to overwrite"):
                write_json_atomic(path, {"first": False})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"first": True})

    def test_atomic_writer_rejects_symlinked_parent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            real.mkdir()
            alias = root / "alias"
            alias.symlink_to(real, target_is_directory=True)
            with self.assertRaises(OSError):
                write_json_atomic(alias / "manifest.json", {"sealed": True})
            self.assertFalse((real / "manifest.json").exists())

    def test_manifest_tampering_breaks_reconstruction(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            manifest = construct_manifest(project.config)
            manifest["partition"]["global_exclusion_targets"].append("injected")
            write_json_atomic(project.output, manifest)
            with self.assertRaisesRegex(JurkatManifestError, "does not reconstruct exactly"):
                authenticate_manifest(project.config, project.output)

    def test_semantically_equal_reformatted_manifest_fails_byte_authentication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            manifest = construct_manifest(project.config)
            project.output.write_text(
                json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
            )
            with self.assertRaisesRegex(JurkatManifestError, "canonical on-disk encoding"):
                authenticate_manifest(project.config, project.output)

    def test_preflight_tampering_and_unregistered_path_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            project = SyntheticProject(Path(temporary))
            manifest = construct_manifest(project.config)
            write_json_atomic(project.output, manifest)
            preflight = construct_training_preflight(project.config, project.output)
            write_json_atomic(project.preflight, preflight)

            alternate = project.preflight.with_name("alternate.json")
            alternate.write_bytes(project.preflight.read_bytes())
            with self.assertRaisesRegex(JurkatManifestError, "path is not registered"):
                authenticate_training_preflight(
                    project.config,
                    project.output,
                    alternate,
                )

            tampered = json.loads(project.preflight.read_text(encoding="utf-8"))
            tampered["datasets"][0]["allowed_source_row_indices_sha256"] = "0" * 64
            project.preflight.write_text(
                json.dumps(tampered, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(JurkatManifestError, "does not reconstruct exactly"):
                authenticate_training_preflight(
                    project.config,
                    project.output,
                    project.preflight,
                )

    def _registered_support(self, path: Path, targets: list[str]) -> dict[str, object]:
        return {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "target_count": len(set(targets) - {"non-targeting"}),
            "target_list_sha256": hashlib.sha256(
                json.dumps(
                    sorted(set(targets) - {"non-targeting"}),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode()
            ).hexdigest(),
        }

    def test_hdf5_soft_and_external_links_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            soft = root / "soft.h5"
            with h5py.File(soft, "w") as data:
                obs = data.create_group("obs")
                write_string_dataset(obs, "actual", ["target", "non-targeting"])
                obs["target_gene"] = h5py.SoftLink("/obs/actual")
            registered = self._registered_support(soft, ["target", "non-targeting"])
            with self.assertRaisesRegex(JurkatManifestError, "not a hard link"):
                read_state_support_targets([(soft, registered)])

            external_data = root / "external-data.h5"
            with h5py.File(external_data, "w") as data:
                write_string_dataset(data, "labels", ["target", "non-targeting"])
            external = root / "external.h5"
            with h5py.File(external, "w") as data:
                obs = data.create_group("obs")
                obs["target_gene"] = h5py.ExternalLink(external_data.name, "/labels")
            registered = self._registered_support(external, ["target", "non-targeting"])
            with self.assertRaisesRegex(JurkatManifestError, "not a hard link"):
                read_state_support_targets([(external, registered)])

            categorical = root / "categorical-soft.h5"
            with h5py.File(categorical, "w") as data:
                obs = data.create_group("obs")
                target = obs.create_group("target_gene")
                target.create_dataset("codes", data=np.asarray([0, 1], dtype=np.int8))
                write_string_dataset(obs, "actual_categories", ["target", "non-targeting"])
                target["categories"] = h5py.SoftLink("/obs/actual_categories")
            registered = self._registered_support(
                categorical, ["target", "non-targeting"]
            )
            with self.assertRaisesRegex(JurkatManifestError, "not a hard link"):
                read_state_support_targets([(categorical, registered)])

            categorical_external = root / "categorical-external.h5"
            with h5py.File(categorical_external, "w") as data:
                obs = data.create_group("obs")
                target = obs.create_group("target_gene")
                target.create_dataset("codes", data=np.asarray([0, 1], dtype=np.int8))
                target["categories"] = h5py.ExternalLink(
                    external_data.name,
                    "/labels",
                )
            registered = self._registered_support(
                categorical_external,
                ["target", "non-targeting"],
            )
            with self.assertRaisesRegex(JurkatManifestError, "not a hard link"):
                read_state_support_targets([(categorical_external, registered)])

    def test_hdf5_virtual_and_external_storage_metadata_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            virtual_source = root / "vsource.h5"
            with h5py.File(virtual_source, "w") as data:
                data.create_dataset("labels", data=np.asarray([1, 2], dtype=np.int32))
            virtual = root / "virtual.h5"
            layout = h5py.VirtualLayout(shape=(2,), dtype=np.int32)
            layout[:] = h5py.VirtualSource(virtual_source, "labels", shape=(2,))
            with h5py.File(virtual, "w", libver="latest") as data:
                obs = data.create_group("obs")
                obs.create_virtual_dataset("target_gene", layout)
            registered = self._registered_support(virtual, ["1", "2"])
            with self.assertRaisesRegex(JurkatManifestError, "is virtual"):
                read_state_support_targets([(virtual, registered)])

            external = root / "external-storage.h5"
            raw_storage = root / "labels.bin"
            with h5py.File(external, "w") as data:
                obs = data.create_group("obs")
                obs.create_dataset(
                    "target_gene",
                    shape=(2,),
                    dtype="S8",
                    external=[(raw_storage.name, 0, h5py.h5f.UNLIMITED)],
                )
            registered = self._registered_support(external, ["target", "other"])
            with self.assertRaisesRegex(JurkatManifestError, "external storage"):
                read_state_support_targets([(external, registered)])

            category_virtual = root / "category-virtual.h5"
            category_layout = h5py.VirtualLayout(shape=(2,), dtype=np.int32)
            category_layout[:] = h5py.VirtualSource(
                virtual_source,
                "labels",
                shape=(2,),
            )
            with h5py.File(category_virtual, "w", libver="latest") as data:
                obs = data.create_group("obs")
                target = obs.create_group("target_gene")
                target.create_dataset("codes", data=np.asarray([0, 1], dtype=np.int8))
                target.create_virtual_dataset("categories", category_layout)
            registered = self._registered_support(category_virtual, ["1", "2"])
            with self.assertRaisesRegex(JurkatManifestError, "is virtual"):
                read_state_support_targets([(category_virtual, registered)])


if __name__ == "__main__":
    unittest.main()
