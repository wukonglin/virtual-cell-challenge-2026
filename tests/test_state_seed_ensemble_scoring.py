from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import authenticate_state_seed_ensemble_for_scoring as prescore  # noqa: E402
import ensemble_state_seed_candidates as ensemble  # noqa: E402
import validate_public_state_seed_ensemble_cell_eval_run as score_validator  # noqa: E402


def descriptor(path: Path) -> dict:
    return prescore._descriptor(path)


def make_plan(tags: list[str]) -> dict:
    output_tag = prescore.REGISTERED_OUTPUT_TAG
    return {
        "schema": ensemble.PLAN_SCHEMA,
        "lineage": ensemble.PLAN_LINEAGE,
        "context": "HepG2",
        "output_tag": output_tag,
        "source_tags": tags,
        "axes": {"contexts": 1, "targets": 300, "cells_per_target": 400, "genes": 18533},
        "method": {
            "name": ensemble.METHOD_NAME,
            "weights": "uniform",
            "integerization": ensemble.INTEGERIZATION,
            "tie_break": ensemble.TIE_BREAK,
            "tie_break_seed": 20260901,
        },
        "evaluation_policy": {
            "absolute_tolerance": 1e-12,
            "best_individual_seed_selection_allowed": False,
            "fallback_tag": tags[0],
            "held_target_gate": "all_evaluable_metrics_non_regression",
            "individual_scores_role": "seed_variance_diagnostic_only",
            "primary_baseline_tag": tags[0],
            "primary_candidate_tag": output_tag,
            "primary_gate": (
                "all_and_direct_mse_nmae_reach_non_regression_and_"
                "pds_or_jaccard_strict_improvement"
            ),
            "public_score_opens_after_ensemble_materialization": True,
        },
        "selection_policy": {
            "all_registered_sources_required": True,
            "best_public_score_seed_selection_allowed": False,
            "failed_integrity_source_exclusion_allowed": False,
            "missing_source_fallback_allowed": False,
        },
        "firewall": {
            "challenge_leaderboard_used_for_selection": False,
            "sealed_treated_profiles_read": False,
            "source_selection_uses_metrics": False,
        },
    }


class Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.project = root / "project"
        self.public = self.project / "artifacts" / "anvil_state_reconstruction"
        self.tags = list(prescore.REGISTERED_SOURCE_TAGS)
        self.plan_payload = make_plan(self.tags)
        self.plan = (
            self.project / "configs" / "state" / "anvil_seed_ensemble_v1.json"
        )
        self.manifest = self.project / "dataset" / "public_v51" / "split_manifest.csv"
        self.ensemble_run = (
            self.public / "ensembles" / "hepg2" / self.plan_payload["output_tag"]
        )
        self.launcher = (
            self.project / "slurm" / "cpu_score_public_state_seed_ensemble_v1.sbatch"
        )
        self.builder = self.project / "scripts" / "ensemble_state_seed_candidates.py"
        for directory in (
            self.plan.parent,
            self.manifest.parent,
            self.ensemble_run,
            self.launcher.parent,
            self.builder.parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self.plan.write_text(json.dumps(self.plan_payload), encoding="utf-8")
        self.manifest.write_text("target_gene,validation_route\ng0,direct\ng1,held_target\n",
                                 encoding="utf-8")
        self.launcher.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
        self.builder.write_text("# fixture builder\n", encoding="utf-8")
        self.prediction = self.ensemble_run / "prediction.h5ad"
        self.prediction.write_bytes(b"derived-prediction")
        self.sources: list[dict] = []
        for ordinal, tag in enumerate(self.tags):
            run = self.public / "candidates" / "hepg2" / tag
            run.mkdir(parents=True)
            paths = {
                "spec": run / "spec.json",
                "generation": run / "generation.json",
                "verification": run / "generation_verified.json",
                "prediction": run / "prediction.h5ad",
                "transport": run / "candidate_transport_verified.json",
            }
            paths["spec"].write_text(
                json.dumps({"inputs": {"panel_csv": descriptor(self.manifest)}}),
                encoding="utf-8",
            )
            for key in ("generation", "verification", "transport"):
                paths[key].write_text(json.dumps({"ordinal": ordinal, "kind": key}),
                                      encoding="utf-8")
            paths["prediction"].write_bytes(f"prediction-{ordinal}".encode())
            self.sources.append(
                {
                    "tag": tag,
                    "run_directory": str(run.resolve()),
                    "artifacts": {key: descriptor(path) for key, path in paths.items()},
                }
            )
        self.ensemble_receipt_payload = {
            "schema": ensemble.RECEIPT_SCHEMA,
            "status": "passed",
            "lineage": "truth-blind-derived-seed-ensemble-not-historical-p4",
            "context": "HepG2",
            "output_tag": self.plan_payload["output_tag"],
            "evaluation_policy": self.plan_payload["evaluation_policy"],
            "sources": self.sources,
            "output": descriptor(self.prediction),
        }
        self.ensemble_receipt = self.ensemble_run / "ensemble_verified.json"
        self.ensemble_receipt.write_text(
            json.dumps(self.ensemble_receipt_payload), encoding="utf-8"
        )
        self.prescore_path = (
            self.public
            / "ensemble_scoring"
            / "hepg2"
            / self.plan_payload["output_tag"]
            / "ensemble_prescore_authenticated.json"
        )

    def build_prescore(self) -> dict:
        with mock.patch.object(
            prescore, "verify_ensemble", return_value=self.ensemble_receipt_payload
        ):
            payload = prescore.build_payload(
                project_dir=self.project,
                public_root=self.public,
                plan_path=self.plan,
                ensemble_run_dir=self.ensemble_run,
                launcher_path=self.launcher,
            )
        self.prescore_path.parent.mkdir(parents=True, exist_ok=True)
        self.prescore_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload


class PrescoreAuthenticationTests(unittest.TestCase):
    def test_build_and_revalidate_all_four_sources(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            payload = fixture.build_prescore()
            observed = prescore.validate_receipt(
                fixture.prescore_path,
                expected_context="HepG2",
                expected_output_tag=fixture.plan_payload["output_tag"],
                plan_path=fixture.plan,
                ensemble_run_dir=fixture.ensemble_run,
                rehash_sources=True,
            )
            self.assertEqual(len(observed["source_candidates"]), 4)
            self.assertEqual(
                observed["identity_kind"], "derived-uniform-state-seed-ensemble"
            )
            self.assertFalse(observed["contract"]["leaderboard_submission_authorized"])
            self.assertEqual(
                payload["evaluation_policy"], fixture.plan_payload["evaluation_policy"]
            )

    def test_source_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            fixture.build_prescore()
            source_prediction = Path(
                fixture.sources[2]["artifacts"]["prediction"]["path"]
            )
            source_prediction.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "descriptor content differs"):
                prescore.validate_receipt(
                    fixture.prescore_path,
                    expected_context="HepG2",
                    expected_output_tag=fixture.plan_payload["output_tag"],
                    plan_path=fixture.plan,
                    ensemble_run_dir=fixture.ensemble_run,
                    rehash_sources=True,
                )

    def test_single_checkpoint_relabel_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            payload = fixture.build_prescore()
            payload["identity_kind"] = "strict-v2-single-checkpoint"
            fixture.prescore_path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "mislabels"):
                prescore.validate_receipt(
                    fixture.prescore_path,
                    expected_context="HepG2",
                    expected_output_tag=fixture.plan_payload["output_tag"],
                    plan_path=fixture.plan,
                    ensemble_run_dir=fixture.ensemble_run,
                    rehash_sources=False,
                )

    def test_registered_plan_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            changed = dict(fixture.plan_payload)
            changed["method"] = dict(changed["method"])
            changed["method"]["tie_break_seed"] += 1
            with self.assertRaisesRegex(RuntimeError, "registered identity"):
                prescore._registered_plan(changed)

    def test_registered_source_order_change_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            changed = dict(fixture.plan_payload)
            changed["source_tags"] = list(reversed(changed["source_tags"]))
            changed["evaluation_policy"] = dict(changed["evaluation_policy"])
            changed["evaluation_policy"]["fallback_tag"] = changed["source_tags"][0]
            changed["evaluation_policy"]["primary_baseline_tag"] = changed["source_tags"][0]
            with self.assertRaisesRegex(RuntimeError, "registered identity"):
                prescore._registered_plan(changed)


class EnsembleScoreReceiptTests(unittest.TestCase):
    def _write_views(self, fixture: Fixture) -> tuple[Path, Path]:
        score_dir = fixture.prescore_path.parent
        prediction_view = score_dir / "prediction_view.h5ad"
        truth_view = score_dir / "truth_view.h5ad"
        prediction_view.write_bytes(b"prediction-view")
        truth_view.write_bytes(b"truth-view")
        views = score_dir / "views.json"
        views.write_text(
            json.dumps(
                {
                    "schema": "vcc-public-scoring-views-v1",
                    "context": "HepG2",
                    "axes": {
                        "genes": 7107,
                        "targets": 300,
                        "prediction_treated_cells": 120000,
                    },
                    "contract": {
                        "model_read_treated_truth": False,
                        "input_type": "raw-nonnegative-integer-counts",
                    },
                    "provenance": {
                        "common_genes": {},
                        "common_genes_json": {},
                        "controls_only": {},
                        "output_prediction": score_validator.describe_file(prediction_view),
                        "output_truth": score_validator.describe_file(truth_view),
                        "prediction": score_validator.describe_file(fixture.prediction),
                        "sealed_truth": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        run_meta = score_dir / "run_meta.json"
        run_meta.write_text(json.dumps({"source": str(truth_view.resolve())}),
                            encoding="utf-8")
        return views, run_meta

    def test_views_bind_derived_prediction_and_prescore_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            fixture.build_prescore()
            views, run_meta = self._write_views(fixture)
            observed = score_validator.validate_ensemble_receipts(
                context="HepG2",
                output_tag=fixture.plan_payload["output_tag"],
                manifest_path=fixture.manifest,
                plan_path=fixture.plan,
                ensemble_verification_path=fixture.ensemble_receipt,
                prescore_path=fixture.prescore_path,
                ensemble_run_dir=fixture.ensemble_run,
                views_path=views,
                run_meta_path=run_meta,
            )
            self.assertEqual(
                observed["source_bundle_identity_sha256"],
                prescore._bundle_digest(fixture.sources),
            )

    def test_views_cannot_substitute_a_different_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as text:
            fixture = Fixture(Path(text))
            fixture.build_prescore()
            views, run_meta = self._write_views(fixture)
            payload = json.loads(views.read_text(encoding="utf-8"))
            alternate = views.parent / "alternate.h5ad"
            alternate.write_bytes(b"alternate")
            payload["provenance"]["prediction"] = score_validator.describe_file(alternate)
            views.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Scoring-view ensemble prediction"):
                score_validator.validate_ensemble_receipts(
                    context="HepG2",
                    output_tag=fixture.plan_payload["output_tag"],
                    manifest_path=fixture.manifest,
                    plan_path=fixture.plan,
                    ensemble_verification_path=fixture.ensemble_receipt,
                    prescore_path=fixture.prescore_path,
                    ensemble_run_dir=fixture.ensemble_run,
                    views_path=views,
                    run_meta_path=run_meta,
                )


class LauncherOrderingTests(unittest.TestCase):
    def test_authentication_precedes_introduction_of_treated_path(self) -> None:
        launcher = (
            ROOT / "slurm" / "cpu_score_public_state_seed_ensemble_v1.sbatch"
        ).read_text(encoding="utf-8")
        authenticate_at = launcher.index('"$CONTRACT_PYTHON" "$PRESCORE_SCRIPT" build')
        treated_path_at = launcher.index("readonly SEALED_TRUTH=")
        truth_hash_at = launcher.index('recorded.get("sha256") != digest.hexdigest()')
        views_at = launcher.index('"$SCORE_PYTHON" scripts/prepare_public_scoring_views.py')
        self.assertLess(authenticate_at, treated_path_at)
        self.assertLess(treated_path_at, truth_hash_at)
        self.assertLess(truth_hash_at, views_at)
        self.assertLess(treated_path_at, views_at)
        self.assertNotIn("hepg2_sealed_truth.h5ad", launcher[:treated_path_at])

    def test_launcher_uses_only_derived_ensemble_contract(self) -> None:
        launcher = (
            ROOT / "slurm" / "cpu_score_public_state_seed_ensemble_v1.sbatch"
        ).read_text(encoding="utf-8")
        self.assertIn("ensemble_prescore_authenticated.json", launcher)
        self.assertIn("validate_public_state_seed_ensemble_cell_eval_run.py", launcher)
        self.assertIn("/ensemble_scoring/", launcher)
        self.assertNotIn("spec.json", launcher)
        self.assertNotIn("public_candidate_v6_contract.py", launcher)

    def test_launcher_pins_paths_runtimes_and_truth_identity(self) -> None:
        launcher = (
            ROOT / "slurm" / "cpu_score_public_state_seed_ensemble_v1.sbatch"
        ).read_text(encoding="utf-8")
        self.assertNotIn("VCC_PUBLIC_ROOT", launcher)
        self.assertNotIn("VCC_SCORE_PYTHON", launcher)
        self.assertNotIn("VCC_CONTRACT_PYTHON", launcher)
        self.assertIn('[[ -e "$SCORE_DIR" || -L "$SCORE_DIR" ]]', launcher)
        self.assertIn('[[ -L "$PUBLIC_ROOT/ensemble_scoring"', launcher)
        self.assertIn('readonly DATA_RECEIPT=', launcher)
        self.assertIn('recorded.get("size_bytes")', launcher)
        self.assertIn('recorded.get("sha256") != digest.hexdigest()', launcher)


if __name__ == "__main__":
    unittest.main()
