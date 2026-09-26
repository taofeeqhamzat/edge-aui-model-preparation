"""
test_trace_ingestion.py
Unit tests verifying the trace ingestion layer for ExperimentTrace v1.1.0 (Task 9.1).
"""

import os
import sys
import json
import filecmp
import tempfile
import unittest
from pathlib import Path
import pyarrow.parquet as pq

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import CANONICAL_EVENT_SCHEMA, find_project_root
from trace_ingestion import (
    parse_experiment_trace,
    ingest_trace_directory,
    EXPERIMENT_TRACE_SCHEMA_VERSION,
    PREPROCESSING_VERSION,
    FEATURE_SCHEMA_VERSION
)
from preprocessing import compute_window_microtensor


def create_minimal_trace_dict(
    session_id: str = "test-session-001",
    experiment_id: str = "exp-alpha",
    condition_id: str = "adaptive",
    task_id: str = "T1",
    num_events: int = 5
) -> dict:
    """Helper creating a strictly valid SerializableExperimentTrace dict."""
    events = []
    base_ts = 1000.0
    for i in range(num_events):
        events.append({
            "timestamp": base_ts + i * 50.0,
            "type": "mousemove" if i % 2 == 0 else "mouseover",
            "x": 0.25 + i * 0.05,
            "y": 0.30 + i * 0.04,
            "componentId": f"comp-{i}",
            "route": "TestRoute",
            "taskId": task_id,
            "viewport": {"width": 1920, "height": 1080},
            "document": {"width": 1920, "height": 2400, "scrollableWidth": 0, "scrollableHeight": 1320}
        })

    return {
        "schemaVersion": EXPERIMENT_TRACE_SCHEMA_VERSION,
        "exportedAt": "2026-09-26T21:00:00.000Z",
        "session": {
            "sessionId": session_id,
            "startedAt": base_ts,
            "experimentId": experiment_id,
            "conditionId": condition_id
        },
        "task": {
            "currentTaskId": task_id,
            "status": "In Progress"
        },
        "metadata": {
            "durationMs": num_events * 50,
            "totalEvents": num_events,
            "conditionId": condition_id
        },
        "behaviourEvents": events,
        "microTensors": [
            {
                "windowId": 0,
                "windowStart": 1000.0,
                "windowEnd": 1500.0,
                "values": [0.0] * 18,
                "eventCount": num_events
            }
        ],
        "macroInteractions": [],
        "outcomes": [],
        "predictions": [],
        "interventions": [],
        "taskEvents": []
    }


class TestTraceIngestion(unittest.TestCase):
    """Test suite covering Task 9.1 trace ingestion requirements."""

    def test_canonical_schema_provenance_fields(self):
        """Verify CANONICAL_EVENT_SCHEMA contains all Task 9.1 provenance fields."""
        self.assertIsNotNone(CANONICAL_EVENT_SCHEMA)
        field_names = [f.name for f in CANONICAL_EVENT_SCHEMA]

        expected_fields = [
            # Snake case
            "session_id", "experiment_id", "condition_id", "task_id", "window_id",
            "source_event_ids", "preprocessing_version", "feature_schema_version",
            # CamelCase aliases
            "sessionId", "experimentId", "conditionId", "taskId", "windowId",
            "sourceEventIds", "preprocessingVersion", "featureSchemaVersion"
        ]
        for field in expected_fields:
            self.assertIn(field, field_names, f"Expected {field} in CANONICAL_EVENT_SCHEMA")

    def test_parse_valid_trace(self):
        """Parse a valid trace dict and assert canonical row mapping and provenance fields."""
        trace = create_minimal_trace_dict()
        rows = parse_experiment_trace(trace)
        self.assertEqual(len(rows), 5)

        for row in rows:
            # Acceptance criteria: Every output row carries:
            # sessionId, experimentId, conditionId, taskId, windowId, sourceEventIds,
            # preprocessingVersion and featureSchemaVersion
            self.assertEqual(row["sessionId"], "test-session-001")
            self.assertEqual(row["experimentId"], "exp-alpha")
            self.assertEqual(row["conditionId"], "adaptive")
            self.assertEqual(row["taskId"], "T1")
            self.assertIsInstance(row["windowId"], int)
            self.assertTrue(row["sourceEventIds"].startswith("test-session-001:ev_"))
            self.assertEqual(row["preprocessingVersion"], PREPROCESSING_VERSION)
            self.assertEqual(row["featureSchemaVersion"], FEATURE_SCHEMA_VERSION)

            # Assert coordinates and viewport
            self.assertGreaterEqual(row["x_norm"], 0.0)
            self.assertLessEqual(row["x_norm"], 1.0)
            self.assertEqual(row["viewport_width"], 1920.0)
            self.assertEqual(row["viewport_height"], 1080.0)

    def test_unmapped_top_level_field_rejected(self):
        """Verify that an unmapped top-level field raises an explicit ValueError naming the field."""
        trace = create_minimal_trace_dict()
        trace["unrecognizedAlienField"] = "bad_payload"

        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Unmapped trace field", str(ctx.exception))
        self.assertIn("unrecognizedAlienField", str(ctx.exception))

    def test_unmapped_event_field_rejected(self):
        """Verify that an unmapped event-level field raises an explicit ValueError naming the field."""
        trace = create_minimal_trace_dict()
        trace["behaviourEvents"][0]["unexpectedSensorReading"] = 42.0

        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Unmapped trace event field", str(ctx.exception))
        self.assertIn("unexpectedSensorReading", str(ctx.exception))

    def test_invalid_schema_version_rejected(self):
        """Verify wrong schemaVersion is rejected."""
        trace = create_minimal_trace_dict()
        trace["schemaVersion"] = "0.9.0"

        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Invalid schemaVersion", str(ctx.exception))

    def test_non_positive_viewport_rejected(self):
        """Verify non-positive viewport dimensions fail visibly."""
        trace = create_minimal_trace_dict()
        trace["behaviourEvents"][0]["viewport"] = {"width": 0, "height": 1080}

        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Non-positive or invalid viewport", str(ctx.exception))

        trace["behaviourEvents"][0]["viewport"] = {"width": 1920, "height": -50}
        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Non-positive or invalid viewport", str(ctx.exception))

    def test_unmodelled_modality_yields_cleared_mask(self):
        """Assert that an unmodelled modality produces a cleared mask bit without zero imputation."""
        events = [
            {"timestamp_ms": 1000, "event_type": "mousemove", "x_norm": 0.5, "y_norm": 0.5},
            {"timestamp_ms": 1100, "event_type": "mousemove", "x_norm": 0.6, "y_norm": 0.6},
            {"timestamp_ms": 1200, "event_type": "mousemove", "x_norm": 0.7, "y_norm": 0.7}
        ]

        # Extract MicroTensor with NO scroll support (unmodelled modality)
        tensor = compute_window_microtensor(
            events=events,
            viewport=(1920, 1080),
            document=(1920, 2000),
            has_pointer_support=True,
            has_dom_support=True,
            has_scroll_support=False  # Unmodelled modality
        )

        # Modality mask indices: features are [0..8], masks are [9..17]
        # Scroll features are 7 (scrollDepthPercentage) and 8 (scrollVelocity)
        # Masks are 9 + 7 = 16 and 9 + 8 = 17
        scroll_depth_mask = tensor[16]
        scroll_vel_mask = tensor[17]

        self.assertEqual(scroll_depth_mask, 0.0, "Unmodelled scroll depth modality mask must be 0.0")
        self.assertEqual(scroll_vel_mask, 0.0, "Unmodelled scroll velocity modality mask must be 0.0")
        self.assertEqual(tensor[7], 0.0, "Unmodelled scroll depth value masked to 0.0")
        self.assertEqual(tensor[8], 0.0, "Unmodelled scroll velocity value masked to 0.0")

    def test_byte_identical_export(self):
        """Assert that two consecutive conversions of the same trace directory produce byte-identical Parquet files."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_dir = Path(tmp_dir) / "traces"
            trace_dir.mkdir()

            # Create 2 trace files
            for i in range(2):
                t = create_minimal_trace_dict(session_id=f"sess-{i}", num_events=6)
                with open(trace_dir / f"trace_{i}.json", "w", encoding="utf-8") as f:
                    json.dump(t, f)

            out1 = Path(tmp_dir) / "run1.parquet"
            out2 = Path(tmp_dir) / "run2.parquet"

            ingest_trace_directory(trace_dir, output_path=out1, force=True)
            ingest_trace_directory(trace_dir, output_path=out2, force=True)

            self.assertTrue(out1.is_file())
            self.assertTrue(out2.is_file())
            self.assertGreater(out1.stat().st_size, 0)
            self.assertTrue(
                filecmp.cmp(str(out1), str(out2), shallow=False),
                "Consecutive runs over identical trace inputs must be byte-identical"
            )

    def test_real_exported_trace_ingestion(self):
        """Test ingestion on real exported traces from edge-aui-framework if present."""
        root = find_project_root()
        framework_traces = root.parent / "edge-aui-framework" / "docs" / "experiments" / "random" / "traces"

        if not framework_traces.is_dir():
            self.skipTest(f"Trace directory not found: {framework_traces}")

        json_traces = list(framework_traces.glob("*.json"))
        if not json_traces:
            self.skipTest(f"No traces found in {framework_traces}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "real_traces.parquet"
            result = ingest_trace_directory(framework_traces, output_path=out_p, force=True)
            self.assertTrue(os.path.isfile(result))

            table = pq.read_table(result)
            self.assertGreater(table.num_rows, 0)
            df = table.to_pandas()

            # Assert all required columns exist and are non-empty
            for col in ["sessionId", "experimentId", "conditionId", "taskId", "windowId", "sourceEventIds"]:
                self.assertIn(col, df.columns)
                self.assertTrue(df[col].notna().any())


if __name__ == "__main__":
    unittest.main()
