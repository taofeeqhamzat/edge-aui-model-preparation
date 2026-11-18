"""
test_window_horizon.py
Task 40.3: small, pre-declared window/horizon experiment, session-bounded splits,
class balance, task coverage and selection.

This is an engineering experiment on scripted captures. It reports dataset/label feasibility
and majority-baseline metrics; it is not participant evidence and it does not train a head.
"""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from window_horizon import (
    EXPERIMENTS,
    MIN_CLASS_SUPPORT,
    render_markdown,
    run_experiment,
    select_config,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / ".data" / "raw" / "scripted"


class TestExperimentMatrix(unittest.TestCase):
    def test_matrix_is_small_and_pre_declared(self):
        self.assertEqual([config["id"] for config in EXPERIMENTS], ["A", "B", "C", "D"])
        by_id = {config["id"]: config for config in EXPERIMENTS}
        self.assertEqual(by_id["A"]["history_windows"], 8)
        self.assertEqual(by_id["A"]["lookahead_ms"], (500, 1500))
        self.assertEqual(by_id["B"]["lookahead_ms"], (1000, 2000))
        self.assertEqual(by_id["C"]["history_windows"], 4)
        self.assertEqual(by_id["D"]["history_windows"], 12)
        self.assertTrue(all(config["stride_ms"] == 250 for config in EXPERIMENTS))
        self.assertGreater(MIN_CLASS_SUPPORT, 0)

    def test_selection_follows_the_pre_declared_criteria(self):
        def result(config_id, coverage, share, lead, order):
            return {
                "config": {"id": config_id},
                "order": order,
                "metrics": {
                    "train_classes_with_min_support": coverage,
                    "train_max_class_share": share,
                    "lead_time_median_ms": lead,
                },
            }

        results = [
            result("A", 3, 0.80, 900.0, 0),
            result("B", 5, 0.60, 1200.0, 1),
            result("C", 5, 0.55, 1300.0, 2),
            result("D", 5, 0.55, 800.0, 3),
        ]
        selected = select_config(results)
        # Highest coverage, then most balanced, then earliest lead time.
        self.assertEqual(selected["config"]["id"], "D")

    def test_selection_is_deterministic_on_a_tie(self):
        def result(config_id, order):
            return {
                "config": {"id": config_id},
                "order": order,
                "metrics": {
                    "train_classes_with_min_support": 4,
                    "train_max_class_share": 0.5,
                    "lead_time_median_ms": 1000.0,
                },
            }
        first = select_config([result("A", 0), result("B", 1)])
        second = select_config([result("B", 1), result("A", 0)])
        self.assertEqual(first["config"]["id"], second["config"]["id"])
        self.assertEqual(first["config"]["id"], "A")


class TestExperimentRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRACE_DIR.is_dir():
            raise unittest.SkipTest(f"Trace directory not found: {TRACE_DIR}")
        cls.report = run_experiment(TRACE_DIR)
        cls.results = cls.report["experiments"]
        cls.selection = cls.report["selection"]

    def test_all_four_configs_are_evaluated(self):
        self.assertEqual([result["config"]["id"] for result in self.results], ["A", "B", "C", "D"])
        for result in self.results:
            self.assertGreater(result["metrics"]["total_examples"], 0, result["config"]["id"])
            self.assertIn(result["config"]["id"], {"A", "B", "C", "D"})

    def test_metrics_cover_the_acceptance_requirements(self):
        required = {
            "per_outcome_frequency",
            "time_between_actions_ms",
            "lead_time_median_ms",
            "per_class_prf",
            "macro_f1",
            "train_max_class_share",
            "false_early_predictions",
            "excluded_incomplete_lookahead",
            "excluded_recorded_only",
            "task_coverage",
            "train_classes_with_min_support",
        }
        for result in self.results:
            self.assertTrue(required.issubset(result["metrics"]), result["config"]["id"])

    def test_false_early_predictions_are_impossible_by_construction(self):
        for result in self.results:
            self.assertEqual(result["metrics"]["false_early_predictions"], 0, result["config"]["id"])

    def test_task_coverage_is_reported_per_partition(self):
        for result in self.results:
            coverage = result["metrics"]["task_coverage"]
            self.assertEqual(set(coverage), {"train", "val", "test"})
            self.assertGreater(sum(sum(part.values()) for part in coverage.values()),
                               result["metrics"]["total_examples"] - 1)

    def test_selection_records_a_rationale(self):
        self.assertIn(self.selection["config"]["id"], {"A", "B", "C", "D"})
        self.assertTrue(self.selection["rationale"])

    def test_report_renders_the_selection_and_caveats(self):
        markdown = render_markdown(self.report)
        self.assertIn("Window/Horizon Experiment", markdown)
        self.assertIn(f"Selected configuration: {self.selection['config']['id']}", markdown)
        self.assertIn("scripted", markdown.lower())
        self.assertIn("not participant", markdown.lower())


class TestSelectedDatasetWrite(unittest.TestCase):
    def test_selected_dataset_and_manifest_record_the_configuration(self):
        from window_horizon import write_selected_dataset

        if not TRACE_DIR.is_dir():
            raise unittest.SkipTest(f"Trace directory not found: {TRACE_DIR}")

        with tempfile.TemporaryDirectory() as tmp:
            written = write_selected_dataset(
                trace_dir=TRACE_DIR, output_dir=Path(tmp), version="v2.0.0-semantic-test"
            )
            self.assertTrue(written["parquet"].is_file())
            self.assertTrue(written["manifest"].is_file())
            manifest = json.loads(written["manifest"].read_text(encoding="utf-8"))
            selection = json.loads(
                (Path(tmp) / "window_horizon_selection.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["target_semantics"]["lookahead_horizon_ms"],
                list(selection["selection"]["config"]["lookahead_ms"]),
            )
            self.assertIn("target_task_outcome", manifest["examples"][0])


if __name__ == "__main__":
    unittest.main()
