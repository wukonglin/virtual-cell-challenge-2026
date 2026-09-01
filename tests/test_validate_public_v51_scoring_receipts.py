"""Tests for closed-schema V5.1 scoring-receipt authentication."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import validate_public_v51_scoring_receipts as module  # noqa: E402


def descriptor(path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def hash_descriptor(path: Path) -> dict[str, str]:
    payload = path.read_bytes()
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def source_descriptor(path: Path) -> dict[str, Any]:
    record = descriptor(path)
    record["mtime_ns"] = path.stat().st_mtime_ns
    return record


class PublicV51ScoringReceiptTests(unittest.TestCase):
    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def _build_context(self, project: Path, context: str) -> Path:
        slug = module.EXPECTED_CONTEXTS[context]
        context_dir = project / "artifacts/public_v51/scoring" / slug
        context_dir.mkdir(parents=True, exist_ok=True)
        cell_eval_module = project / "external/cell-eval2/src/cell_eval2/__init__.py"
        cell_eval_module.parent.mkdir(parents=True, exist_ok=True)
        cell_eval_module.write_text("", encoding="utf-8")

        runtime_version = "0.0.0+unknown"
        config_digest = "1" * 64
        semantic_identity = "2" * 64
        source_fingerprint = ("3" if context == "HepG2" else "4") * 64

        dataset_root = project / "dataset/public_v51"
        dataset_root.mkdir(parents=True, exist_ok=True)
        source_names = ("common_genes.csv", "common_genes.json")
        for name in source_names:
            path = dataset_root / name
            if not path.exists():
                path.write_bytes(f"fixture-{name}".encode("utf-8"))
        for source_context, source_slug in module.EXPECTED_CONTEXTS.items():
            for suffix in ("controls_only.h5ad", "sealed_truth.h5ad"):
                path = dataset_root / f"{source_slug}_{suffix}"
                if not path.exists():
                    path.write_bytes(
                        f"fixture-{source_context}-{suffix}".encode("utf-8")
                    )

        full_dir = project / "artifacts/public_v51/full"
        full_dir.mkdir(parents=True, exist_ok=True)
        for prediction_context, prediction_slug in module.EXPECTED_CONTEXTS.items():
            for prediction_label in module.EXPECTED_LABELS:
                path = full_dir / f"{prediction_slug}_{prediction_label}.h5ad"
                if not path.exists():
                    path.write_bytes(
                        f"prediction-{prediction_context}-{prediction_label}".encode(
                            "utf-8"
                        )
                    )
        summary_contexts: dict[str, dict[str, Any]] = {}
        for summary_context, summary_slug in module.EXPECTED_CONTEXTS.items():
            anchor_path = full_dir / f"{summary_slug}_anchor.h5ad"
            candidate_path = full_dir / f"{summary_slug}_v51_alpha010.h5ad"
            summary_contexts[summary_context] = {
                "clip_fraction": 1e-6,
                "direct_changed_entries": 10,
                "fallback_changed_entries": 0,
                "anchor_sha256": hash_descriptor(anchor_path)["sha256"],
                "candidate_sha256": hash_descriptor(candidate_path)["sha256"],
            }
        summary_path = full_dir / "public_state_anchor_v51_full_summary.json"
        self._write_json(
            summary_path,
            {"schema": module.FULL_SUMMARY_SCHEMA, "contexts": summary_contexts},
        )

        resolved = {
            label: {
                "config_digest": config_digest,
                "resolved_device": "cpu",
                "resolved_de_backend": "pdex",
                "cell_eval2_version": runtime_version,
                "pdex_version": module.EXPECTED_PDEX_VERSION,
            }
            for label in module.EXPECTED_LABELS
        }
        outputs: dict[str, dict[str, Any]] = {}
        for label in module.EXPECTED_LABELS:
            prediction_path = full_dir / f"{slug}_{label}.h5ad"
            prediction_view_path = context_dir / f"{label}_prediction_view.h5ad"
            truth_path = context_dir / f"{label}_truth_view.h5ad"
            prediction_view_path.write_bytes(
                f"prediction-view-{context}-{label}".encode("utf-8")
            )
            truth_path.write_bytes(f"truth-{context}".encode("utf-8"))
            views_path = context_dir / f"{label}_views.json"
            self._write_json(
                views_path,
                {
                    "schema": module.VIEWS_SCHEMA,
                    "context": context,
                    "axes": {
                        "genes": 7107,
                        "observed_control_cells": 10,
                        "prediction_treated_cells": 120000,
                        "prediction_view_cells": 120010,
                        "targets": 300,
                        "truth_view_cells": 20,
                    },
                    "contract": module.VIEWS_CONTRACT,
                    "provenance": {
                        "common_genes": source_descriptor(
                            dataset_root / "common_genes.csv"
                        ),
                        "common_genes_json": source_descriptor(
                            dataset_root / "common_genes.json"
                        ),
                        "controls_only": source_descriptor(
                            dataset_root / f"{slug}_controls_only.h5ad"
                        ),
                        "sealed_truth": source_descriptor(
                            dataset_root / f"{slug}_sealed_truth.h5ad"
                        ),
                        "prediction": source_descriptor(prediction_path),
                        "output_prediction": descriptor(prediction_view_path),
                        "output_truth": descriptor(truth_path),
                    },
                },
            )
            score_dir = context_dir / f"{label}_cell_eval2"
            score_dir.mkdir(parents=True, exist_ok=True)
            for filename in ("results.csv", "agg_results.csv", "metric_aggregation.csv"):
                (score_dir / filename).write_text(
                    f"fixture,{context},{label},{filename}\n", encoding="utf-8"
                )
            run_meta_path = score_dir / "run_meta.json"
            self._write_json(
                run_meta_path,
                {
                    "config_digest": config_digest,
                    "cell_eval2_version": runtime_version,
                    "resolved_de_backend": "pdex",
                    "resolved_device": "cpu",
                    "environment": {
                        "pdex": {"version": module.EXPECTED_PDEX_VERSION}
                    },
                    "source_fingerprint_strict": True,
                    "source_fingerprint": source_fingerprint,
                    "source": str(truth_path.resolve()),
                    "comparator": "bulk_lognorm",
                    "input_type_real_effective": "counts",
                    "input_type_pred_effective": "counts",
                    "anchor_semantic_identity": semantic_identity,
                },
            )
            outputs[label] = {
                output_name: descriptor(
                    context_dir / relative_template.format(label=label)
                )
                for output_name, relative_template in module.OUTPUT_LAYOUT.items()
            }

        receipt = {
            "schema": module.SCHEMA,
            "status": "scoring_complete",
            "context": context,
            "full_generation_summary": hash_descriptor(summary_path),
            "predictions": {
                label: hash_descriptor(full_dir / f"{slug}_{label}.h5ad")
                for label in module.EXPECTED_LABELS
            },
            "generation_qc": summary_contexts[context],
            "scoring_software": {
                "cell_eval2_git_commit": module.EXPECTED_CELL_EVAL_COMMIT,
                "cell_eval2_checkout_clean": True,
                "cell_eval2_declared_version": module.EXPECTED_CELL_EVAL_VERSION,
                "cell_eval2_runtime_version": runtime_version,
                "cell_eval2_module": str(cell_eval_module.resolve()),
                "pdex_version": module.EXPECTED_PDEX_VERSION,
            },
            "resolved_scoring": resolved,
            "outputs": outputs,
        }
        receipt_path = context_dir / "scoring_receipt.json"
        self._write_json(receipt_path, receipt)
        return receipt_path

    def _load_receipt(self, path: Path) -> dict[str, Any]:
        return json.loads(path.read_text(encoding="utf-8"))

    def test_valid_receipts_authenticate_every_output_and_shared_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            entries = [
                (self._build_context(project, "HepG2"), "HepG2"),
                (self._build_context(project, "Jurkat"), "Jurkat"),
            ]
            report = module.validate_receipts(entries)
            self.assertEqual(report["schema"], module.REPORT_SCHEMA)
            self.assertEqual(set(report["contexts"]), {"HepG2", "Jurkat"})
            self.assertEqual(
                report["shared_scoring_identity"]["config_digest"], "1" * 64
            )
            for context in report["contexts"].values():
                for label in module.EXPECTED_LABELS:
                    self.assertEqual(
                        set(context["outputs"][label]),
                        set(module.OUTPUT_LAYOUT) | {"view_artifacts"},
                    )

    def test_every_stamped_output_is_rehashed(self) -> None:
        for output_name in module.OUTPUT_LAYOUT:
            with self.subTest(output_name=output_name):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    stamped_path = Path(
                        receipt["outputs"]["anchor"][output_name]["path"]
                    )
                    payload = stamped_path.read_bytes()
                    self.assertTrue(payload)
                    stamped_path.write_bytes(
                        bytes([payload[0] ^ 1]) + payload[1:]
                    )
                    with self.assertRaisesRegex(RuntimeError, "stamped file hash mismatch"):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_descriptor_schema_path_size_and_hash_are_fail_closed(self) -> None:
        mutations = {
            "extra field": lambda record: record.update({"mtime_ns": 1}),
            "path": lambda record: record.update({"path": record["path"] + ".other"}),
            "size": lambda record: record.update({"size_bytes": record["size_bytes"] + 1}),
            "boolean size": lambda record: record.update({"size_bytes": True}),
            "hash": lambda record: record.update({"sha256": "0" * 64}),
        }
        expected_messages = {
            "extra field": "descriptor fields differ",
            "path": "descriptor path mismatch",
            "size": "stamped file size mismatch",
            "boolean size": "size_bytes is not a nonnegative integer",
            "hash": "stamped file hash mismatch",
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    mutate(receipt["outputs"]["anchor"]["results"])
                    self._write_json(receipt_path, receipt)
                    with self.assertRaisesRegex(RuntimeError, expected_messages[name]):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_receipt_and_config_identity_fields_are_exact(self) -> None:
        mutations = {
            "status": lambda receipt: receipt.update({"status": "inputs_authenticated"}),
            "top-level extra": lambda receipt: receipt.update({"unexpected": True}),
            "software extra": lambda receipt: receipt["scoring_software"].update(
                {"unexpected": True}
            ),
            "resolved extra": lambda receipt: receipt["resolved_scoring"]["anchor"].update(
                {"unexpected": True}
            ),
            "output extra": lambda receipt: receipt["outputs"]["anchor"].update(
                {"unexpected": {}}
            ),
            "config drift": lambda receipt: receipt["resolved_scoring"][
                "v51_alpha010"
            ].update({"config_digest": "9" * 64}),
        }
        expected_messages = {
            "status": "scoring receipt is incomplete",
            "top-level extra": "scoring receipt fields are not exact",
            "software extra": "scoring_software fields are not exact",
            "resolved extra": "resolved scorer fields are not exact",
            "output extra": "output fields are not exact",
            "config drift": "config identities differ",
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    mutate(receipt)
                    self._write_json(receipt_path, receipt)
                    with self.assertRaisesRegex(RuntimeError, expected_messages[name]):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_run_metadata_is_bound_to_the_receipt_and_truth_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            receipt_path = self._build_context(project, "HepG2")
            receipt = self._load_receipt(receipt_path)
            run_meta_path = Path(
                receipt["outputs"]["anchor"]["run_meta"]["path"]
            )
            run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
            run_meta["source"] = str(run_meta_path.resolve())
            self._write_json(run_meta_path, run_meta)
            receipt["outputs"]["anchor"]["run_meta"] = descriptor(run_meta_path)
            self._write_json(receipt_path, receipt)
            with self.assertRaisesRegex(RuntimeError, "scorer source path mismatch"):
                module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_generation_summary_prediction_and_qc_chain_is_closed(self) -> None:
        mutations = ("summary schema", "generation qc", "prediction bytes")
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    if mutation == "summary schema":
                        summary_path = Path(
                            receipt["full_generation_summary"]["path"]
                        )
                        summary = json.loads(summary_path.read_text(encoding="utf-8"))
                        summary["schema"] = "forged-summary"
                        self._write_json(summary_path, summary)
                        receipt["full_generation_summary"] = hash_descriptor(
                            summary_path
                        )
                        expected = "full-generation summary identity mismatch"
                    elif mutation == "generation qc":
                        receipt["generation_qc"]["direct_changed_entries"] += 1
                        expected = "receipt generation QC differs from the full summary"
                    else:
                        prediction_path = Path(
                            receipt["predictions"]["anchor"]["path"]
                        )
                        payload = prediction_path.read_bytes()
                        prediction_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                        expected = "stamped file hash mismatch"
                    self._write_json(receipt_path, receipt)
                    with self.assertRaisesRegex(RuntimeError, expected):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_both_view_artifacts_are_rehashed(self) -> None:
        for artifact in ("output_prediction", "output_truth"):
            with self.subTest(artifact=artifact):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    views_path = Path(
                        receipt["outputs"]["anchor"]["views_report"]["path"]
                    )
                    views = json.loads(views_path.read_text(encoding="utf-8"))
                    artifact_path = Path(views["provenance"][artifact]["path"])
                    payload = artifact_path.read_bytes()
                    artifact_path.write_bytes(bytes([payload[0] ^ 1]) + payload[1:])
                    with self.assertRaisesRegex(
                        RuntimeError, "stamped file hash mismatch"
                    ):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_view_descriptor_schema_and_prediction_cross_link_fail_closed(self) -> None:
        mutations = {
            "nested extra": lambda provenance: provenance["output_prediction"].update(
                {"unexpected": True}
            ),
            "prediction path": lambda provenance: provenance["prediction"].update(
                {"path": provenance["prediction"]["path"] + ".other"}
            ),
        }
        expected_messages = {
            "nested extra": "descriptor fields differ",
            "prediction path": "descriptor path mismatch",
        }
        for name, mutate in mutations.items():
            with self.subTest(mutation=name):
                with tempfile.TemporaryDirectory() as directory:
                    project = Path(directory)
                    receipt_path = self._build_context(project, "HepG2")
                    receipt = self._load_receipt(receipt_path)
                    views_path = Path(
                        receipt["outputs"]["anchor"]["views_report"]["path"]
                    )
                    views = json.loads(views_path.read_text(encoding="utf-8"))
                    mutate(views["provenance"])
                    self._write_json(views_path, views)
                    receipt["outputs"]["anchor"]["views_report"] = descriptor(
                        views_path
                    )
                    self._write_json(receipt_path, receipt)
                    with self.assertRaisesRegex(
                        RuntimeError, expected_messages[name]
                    ):
                        module.validate_scoring_receipt(receipt_path, "HepG2")

    def test_cross_context_config_identity_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            hepg2_receipt = self._build_context(project, "HepG2")
            jurkat_receipt = self._build_context(project, "Jurkat")
            receipt = self._load_receipt(jurkat_receipt)
            replacement_digest = "9" * 64
            for label in module.EXPECTED_LABELS:
                receipt["resolved_scoring"][label]["config_digest"] = replacement_digest
                run_meta_path = Path(
                    receipt["outputs"][label]["run_meta"]["path"]
                )
                run_meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
                run_meta["config_digest"] = replacement_digest
                self._write_json(run_meta_path, run_meta)
                receipt["outputs"][label]["run_meta"] = descriptor(run_meta_path)
            self._write_json(jurkat_receipt, receipt)
            with self.assertRaisesRegex(
                RuntimeError, "Cross-context scoring identity differs: config_digest"
            ):
                module.validate_receipts(
                    [(hepg2_receipt, "HepG2"), (jurkat_receipt, "Jurkat")]
                )


if __name__ == "__main__":
    unittest.main()
