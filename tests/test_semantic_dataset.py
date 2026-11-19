"""
test_semantic_dataset.py
Task 40.2: semantic dataset assembly, exclusions, provenance and manifest round-trip.

The label semantics themselves are pinned in test_semantic_targets.py against the shared
TypeScript fixtures; this module checks that the dataset pipeline uses them correctly.
"""

import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data_manager import create_dataset_manifest, validate_dataset_manifest
from semantic_targets import (
    DEFAULT_LOOKAHEAD_MS,
    RECORDED_ONLY_OUTCOMES,
    TARGET_NAME_TO_ID,
    TARGET_VOCABULARY,
    TARGET_VOCABULARY_VERSION,
)
from target_dataset import (
    build_target_dataset_from_traces,
    save_target_dataset_parquet,
    split_target_dataset,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / ".data" / "raw" / "scripted"


def _build_semantic_trace(outcomes, n_windows=10, tail_ms=1000.0):
    """Minimal 1.4.0 trace with controllable windows and outcome records."""
    windows = [
        {
            "windowId": index,
            "windowStart": 1000.0 + index * 500.0,
            "windowEnd": 1500.0 + index * 500.0,
            "values": [0.0] * 18,
            "eventCount": 1,
        }
        for index in range(n_windows)
    ]
    events = [
        {
            "timestamp": 1010.0 + index * 500.0,
            "type": "mousemove",
            "x": 0.5,
            "y": 0.5,
            "componentId": f"component-{index}",
            "route": "Analytics",
            "taskId": "TD1",
            "viewport": {"width": 1920, "height": 1080},
            "document": {"width": 1920, "height": 1080},
        }
        for index in range(n_windows)
    ]
    events.append({
        "timestamp": windows[-1]["windowEnd"] + tail_ms,
        "type": "mousemove",
        "x": 0.5,
        "y": 0.5,
        "componentId": "tail",
        "route": "Analytics",
        "taskId": "TD1",
        "viewport": {"width": 1920, "height": 1080},
        "document": {"width": 1920, "height": 1080},
    })
    return {
        "schemaVersion": "1.4.0",
        "exportedAt": "2026-11-18T00:00:00.000Z",
        "session": {"sessionId": "synthetic-1", "experimentId": "exp", "conditionId": "adaptive"},
        "task": {"currentTaskId": "TD1", "status": "In Progress"},
        "metadata": {
            "taskDefinitionVersion": "TD-v2",
            "semanticOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
        },
        "behaviourEvents": events,
        "microTensors": windows,
        "outcomes": outcomes,
        "predictions": [],
        "policyDecisions": [],
        "interventions": [],
        "macroInteractions": [],
        "taskEvents": [],
    }


class TestSemanticDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRACE_DIR.is_dir():
            raise unittest.SkipTest(f"Trace directory not found: {TRACE_DIR}")
        cls.rows, cls.stats = build_target_dataset_from_traces(
            trace_dir=TRACE_DIR, target="semantic"
        )

    def test_semantic_rows_carry_target_vocabulary_and_provenance(self):
        self.assertGreater(len(self.rows), 0)
        for row in self.rows:
            self.assertIn(row["target_task_outcome"], TARGET_NAME_TO_ID)
            self.assertEqual(row["target_task_outcome_id"], TARGET_NAME_TO_ID[row["target_task_outcome"]])
            self.assertEqual(row["target_vocabulary_version"], TARGET_VOCABULARY_VERSION)
            self.assertEqual(row["recorded_vocabulary_version"], "testbed-semantic/1.0.0")
            self.assertEqual(row["task_definition_version"], "TD-v2")
            self.assertIn(row["provenance"], {"scripted", "participant"})
            self.assertIn(
                row["target_label_source"],
                {"trace_semantic_outcome", "trace_no_outcome", "lookahead_no_outcome",
                 "re_derived", "re_derived_no_match"},
            )

    def test_recorded_only_outcome_is_never_a_label(self):
        for row in self.rows:
            self.assertNotIn(row["target_task_outcome"], RECORDED_ONLY_OUTCOMES)

    def test_labels_are_future_and_inside_the_lookahead(self):
        minimum, maximum = DEFAULT_LOOKAHEAD_MS
        observed = [row for row in self.rows if row["lead_time_ms"] is not None]
        self.assertGreater(len(observed), 0)
        for row in observed:
            self.assertGreater(row["lead_time_ms"], 0.0)
            self.assertGreaterEqual(row["lead_time_ms"], minimum)
            self.assertLessEqual(row["lead_time_ms"], maximum)
            self.assertIsNotNone(row["target_event_timestamp"])

    def test_exclusion_counters_are_reported(self):
        for key in ("excluded_incomplete_lookahead", "excluded_recorded_only",
                    "excluded_missing_context", "label_sources"):
            self.assertIn(key, self.stats)
        self.assertGreater(self.stats["excluded_incomplete_lookahead"], 0)
        self.assertEqual(
            sum(self.stats["label_sources"].values()),
            len(self.rows),
            "every retained row must have exactly one label source",
        )

    def test_generic_columns_are_retained_for_ablation(self):
        for row in self.rows:
            self.assertIn(row["target_intervention"], {
                "simplify_options", "highlight_primary_action", "offer_assistance",
                "expand_tooltip", "no_op",
            })
            self.assertIsNotNone(row["target_outcome"])

    def test_parquet_round_trip_keeps_semantic_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic.parquet"
            save_target_dataset_parquet(self.rows, path)
            table = pq.read_table(str(path))
            self.assertIn("target_task_outcome", table.column_names)
            self.assertIn("target_task_outcome_id", table.column_names)
            self.assertIn("lead_time_ms", table.column_names)
            self.assertEqual(table.num_rows, len(self.rows))


class TestSemanticDatasetExclusions(unittest.TestCase):
    def test_inspect_help_window_is_excluded_and_counted(self):
        # Anchor window 7 ends at 5000 ms; its horizon is (5500, 6500]. Put a single
        # inspect_help act at 5600 ms and nothing else inside the horizon.
        outcomes = [{
            "timestamp": 5600.0,
            "windowId": 9,
            "sourceEventTimestamp": 5600.0,
            "sourceEventType": "click",
            "componentId": "tooltip-region",
            "taskOutcome": "inspect_help",
            "taskOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
            "taskOutcomeSource": "application_rule",
            "taskOutcomeRuleId": "inspect_help",
        }]
        trace = _build_semantic_trace(outcomes)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            rows, stats = build_target_dataset_from_traces(
                trace_dir=tmp, target="semantic", seq_len=8
            )
        self.assertEqual(stats["excluded_recorded_only"], 1)
        self.assertNotIn(7, [row["anchor_window_id"] for row in rows])
        self.assertNotIn("inspect_help", [row["target_task_outcome"] for row in rows])

    def test_incomplete_lookahead_is_excluded_not_labelled(self):
        trace = _build_semantic_trace([], n_windows=10, tail_ms=0.0)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            rows, stats = build_target_dataset_from_traces(
                trace_dir=tmp, target="semantic", seq_len=8
            )
        # Every anchor's horizon closes at 6500+ ms, past the session end at 6000 ms.
        self.assertEqual(rows, [])
        self.assertEqual(stats["excluded_incomplete_lookahead"], 3)

    def test_semantic_label_uses_the_earliest_target_act(self):
        outcomes = [
            {
                "timestamp": 5700.0,
                "windowId": 9,
                "sourceEventTimestamp": 5700.0,
                "componentId": "open-report-eu-books",
                "taskOutcome": "select_report",
                "taskOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
                "taskOutcomeSource": "application_rule",
            },
            {
                "timestamp": 6200.0,
                "windowId": 9,
                "sourceEventTimestamp": 6200.0,
                "componentId": "btn-export",
                "taskOutcome": "export_report",
                "taskOutcomeVocabularyVersion": "testbed-semantic/1.0.0",
                "taskOutcomeSource": "application_rule",
            },
        ]
        trace = _build_semantic_trace(outcomes)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.json"
            path.write_text(json.dumps(trace), encoding="utf-8")
            rows, _ = build_target_dataset_from_traces(
                trace_dir=tmp, target="semantic", seq_len=8
            )
        anchor7 = [row for row in rows if row["anchor_window_id"] == 7]
        self.assertEqual(len(anchor7), 1)
        self.assertEqual(anchor7[0]["target_task_outcome"], "select_report")
        self.assertEqual(anchor7[0]["target_task_outcome_id"], TARGET_NAME_TO_ID["select_report"])
        self.assertEqual(anchor7[0]["lead_time_ms"], 700.0)


class TestSemanticSplitsAndManifest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not TRACE_DIR.is_dir():
            raise unittest.SkipTest(f"Trace directory not found: {TRACE_DIR}")
        rows, stats = build_target_dataset_from_traces(trace_dir=TRACE_DIR, target="semantic")
        cls.rows, cls.summary = split_target_dataset(
            copy.deepcopy(rows),
            vocabulary=list(TARGET_VOCABULARY.values()),
            label_key="target_task_outcome",
            id_key="target_task_outcome_id",
            label_name="semantic",
            build_stats=stats,
        )

    def test_semantic_split_is_session_bounded_with_task_coverage(self):
        self.assertEqual(self.summary["split_unit"], "session")
        self.assertEqual(self.summary["label_key"], "target_task_outcome")
        self.assertEqual(self.summary["label_vocabulary"], list(TARGET_VOCABULARY.values()))
        sessions = self.summary["sessions"]
        self.assertFalse(set(sessions["train"]) & set(sessions["val"]))
        self.assertFalse(set(sessions["train"]) & set(sessions["test"]))
        self.assertFalse(set(sessions["val"]) & set(sessions["test"]))
        total_coverage = sum(sum(c.values()) for c in self.summary["task_coverage"].values())
        self.assertEqual(total_coverage, len(self.rows))
        self.assertEqual(self.summary["class_weights"]["derived_on"], "train_partition_only")
        self.assertEqual(len(self.summary["class_weights"]["weights"]), len(TARGET_VOCABULARY))
        self.assertIn("label_sources", self.summary["build_stats"])

    def test_semantic_manifest_records_vocabulary_and_split_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            create_dataset_manifest(
                rows=self.rows,
                split_summary=self.summary,
                trace_dir=TRACE_DIR,
                output_path=manifest_path,
                dataset_version="v2.0.0-semantic",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertIn("target_semantics", manifest)
            self.assertEqual(manifest["target_semantics"]["split_unit"], "session")
            self.assertEqual(
                manifest["target_semantics"]["target_vocabulary"],
                list(TARGET_VOCABULARY.values()),
            )
            example = manifest["examples"][0]
            self.assertIn("target_task_outcome", example)
            self.assertIn("target_label_source", example)
            self.assertIn("lead_time_ms", example)

            result = validate_dataset_manifest(manifest_path, trace_dir=TRACE_DIR)
            self.assertTrue(result["valid"])

    def test_semantic_manifest_rejects_a_label_outside_the_vocabulary(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            create_dataset_manifest(
                rows=self.rows,
                split_summary=self.summary,
                trace_dir=TRACE_DIR,
                output_path=manifest_path,
                dataset_version="v2.0.0-semantic",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["examples"][0]["target_task_outcome"] = "inspect_help"
            with self.assertRaises(Exception) as ctx:
                validate_dataset_manifest(manifest, trace_dir=TRACE_DIR)
            self.assertIn("not in the declared target vocabulary", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
