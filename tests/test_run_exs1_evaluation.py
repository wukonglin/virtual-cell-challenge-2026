"""Focused evaluation/firewall tests for the EXS-1 command."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import exs1_expression_scale_core as core  # noqa: E402
from exs1_local_io import sha256_file  # noqa: E402
from run_exs1_expression_scale_repair import (  # noqa: E402
    MSE_METRIC,
    NMAE_METRIC,
    PDS_METRIC,
    main,
)


class EXS1EvaluationTests(unittest.TestCase):
    def test_partial_evaluation_keeps_unavailable_scores_null(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            experiment_path = root / "experiment.json"
            output_path = root / "evaluation.json"
            experiment = {
                "schema": core.SCHEMA,
                "status": "transformed_unscored",
                "experiment": "EXS-1",
                "context": "HepG2",
                "registered_gate": core.REGISTERED_GATE,
                "firewall": {
                    "treated_truth_profiles_read": False,
                    "transform_parser_accepts_treated_truth": False,
                },
                "axes": {
                    "targets": 2,
                    "target_order_first_occurrence": [
                        "GENE_A",
                        "GENE_B",
                    ],
                },
                "arms": {
                    "attenuation_5000bp__keep_prediction": {
                        "output_prediction": {}
                    }
                },
            }
            experiment_path.write_text(
                json.dumps(experiment), encoding="utf-8"
            )
            with mock.patch(
                "run_exs1_expression_scale_repair."
                "_authenticate_scorer",
                return_value={"git_commit": "pinned"},
            ):
                main(
                    [
                        "evaluate",
                        "--experiment-json",
                        str(experiment_path),
                        "--experiment-expected-sha256",
                        sha256_file(experiment_path),
                        "--cell-eval-checkout",
                        str(root / "not-opened"),
                        "--output-json",
                        str(output_path),
                        "--allow-partial",
                    ]
                )

            report = json.loads(
                output_path.read_text(encoding="utf-8")
            )
            self.assertEqual(report["status"], "incomplete")
            self.assertIsNone(
                report["gate_result"]["passed"]
            )
            arm = report["arms"][
                "attenuation_5000bp__keep_prediction"
            ]
            self.assertEqual(
                arm["scores"],
                {
                    MSE_METRIC: None,
                    PDS_METRIC: None,
                    NMAE_METRIC: None,
                },
            )
            self.assertIsNone(arm["gate"]["passed"])

    def test_transform_parser_has_no_truth_input(self) -> None:
        parser = core._parser()
        transform = next(
            action
            for action in parser._actions
            if getattr(action, "dest", None) == "command"
        ).choices["transform"]
        option_strings = {
            option
            for action in transform._actions
            for option in action.option_strings
        }
        self.assertNotIn("--truth", option_strings)
        self.assertNotIn("--sealed-truth", option_strings)
        self.assertNotIn("--treated-truth", option_strings)


if __name__ == "__main__":
    unittest.main()
