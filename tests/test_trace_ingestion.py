"""
test_trace_ingestion.py
Unit tests verifying the trace ingestion layer for ExperimentTrace (Task 9.1).
Covers the current 1.3.0 contract plus supported legacy 1.2.0 / 1.1.0 captures.
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
    UnsupportedTraceSchemaError,
    EXPERIMENT_TRACE_SCHEMA_VERSION,
    SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS,
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


def create_minimal_trace_dict_v130(
    session_id: str = "test-session-130",
    provenance: str = "scripted",
    num_events: int = 5
) -> dict:
    """
    Helper creating a strictly valid ExperimentTrace schemaVersion 1.3.0 dict,
    including `policyDecisions` and the new metadata/session fields.
    """
    trace = create_minimal_trace_dict(session_id=session_id, num_events=num_events)
    trace["schemaVersion"] = "1.3.0"

    trace["session"]["provenance"] = provenance
    trace["session"]["participantId"] = None

    trace["metadata"].update({
        "policyDecisionCount": 1,
        "clock": "epoch_ms",
        "provenance": provenance,
        "applicationVersion": "0.1.0",
        "modelVersion": "gru-foundation-v1",
        "executionProvider": "wasm",
        "policyVersion": "1.0.0",
        "mining": {"executed": 1, "skipped": 0, "superseded": 0, "timedOut": 0, "droppedWindows": 0},
        "evictions": {"total": 0, "byBuffer": {}, "truncated": False},
        "integrityWarnings": []
    })

    trace["policyDecisions"] = [{
        "timestamp": 1000.0,
        "sessionId": session_id,
        "windowId": 0,
        "predictionId": "pred-0",
        "accepted": True,
        "policyDecision": "accepted",
        "policyReason": "candidate accepted for adaptive condition",
        "phase": "policy"
    }]

    trace["predictions"] = [{
        "timestamp": 1000.0,
        "predictionId": "pred-0",
        "evaluatedWindowIds": [0],
        "matchedGate": "slow",
        "windowId": 0,
        "bothGatesEvaluated": True
    }]
    return trace


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
            "provenance",
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

    def test_schema_130_trace_accepted(self):
        """A schemaVersion 1.3.0 trace (policyDecisions + new metadata) is accepted and yields rows."""
        trace = create_minimal_trace_dict_v130(provenance="participant")
        self.assertEqual(trace["schemaVersion"], EXPERIMENT_TRACE_SCHEMA_VERSION)

        rows = parse_experiment_trace(trace)
        self.assertEqual(len(rows), 5)

        for row in rows:
            self.assertEqual(row["sessionId"], "test-session-130")
            # `policyDecisions` is accepted-and-ignored for now (deliberately deferred),
            # but its presence must not cause the trace to be rejected.
            self.assertEqual(row["provenance"], "participant")
            self.assertEqual(row["preprocessing_version"], PREPROCESSING_VERSION)
            self.assertEqual(row["feature_schema_version"], FEATURE_SCHEMA_VERSION)

    def test_epoch_millisecond_windows_do_not_overflow_window_id(self):
        """
        Regression: epoch-millisecond timestamps must not produce an out-of-range window id.

        Schema 1.3.0 places every record on one epoch clock, so window bounds and event
        timestamps are ~1.79e12 rather than a short monotonic value. The previous
        window-correlation fallback was `int(ts_ms // 250)`, which against an epoch timestamp
        yields an id in the billions and raises
        `ArrowInvalid: Value ... too large to fit in C integer type` when the canonical Parquet
        is written — aborting the ingestion of an otherwise valid trace.

        This test drives the *real* ingestion entry point and writes real Parquet, because the
        overflow only occurs at that boundary; a row-level assertion would not have caught it.
        """
        epoch_base = 1_791_045_408_000  # 2026-10-01, an epoch-millisecond value
        trace = create_minimal_trace_dict_v130(provenance="scripted", num_events=6)

        # Two adjacent 250 ms windows on the epoch timeline, and events inside them.
        trace["microTensors"] = [
            {
                "windowId": 0,
                "windowStart": float(epoch_base),
                "windowEnd": float(epoch_base + 250),
                "values": [0.1] * 18,
            },
            {
                "windowId": 1,
                "windowStart": float(epoch_base + 250),
                "windowEnd": float(epoch_base + 500),
                "values": [0.1] * 18,
            },
        ]

        event_timestamps = [
            epoch_base + 10,
            epoch_base + 60,
            epoch_base + 200,
            # Exactly on the boundary of window 0; the grid is half-open [start, end).
            epoch_base + 250,
            epoch_base + 300,
            # Past the last window end: must fall back to the latest closing window, not to
            # `ts // 250`.
            epoch_base + 700,
        ]
        for event, ts in zip(trace["behaviourEvents"], event_timestamps):
            event["timestamp"] = ts

        with tempfile.TemporaryDirectory() as tmp:
            trace_dir = Path(tmp) / "traces"
            trace_dir.mkdir()
            (trace_dir / "epoch-windows.json").write_text(json.dumps(trace))

            output = Path(tmp) / "canonical.parquet"
            ingest_trace_directory(str(trace_dir), output_path=str(output), force=True)

            table = pq.read_table(output)

        self.assertEqual(table.num_rows, len(event_timestamps))

        window_ids = table.column("windowId").to_pylist()
        # No fabricated, out-of-range window ids, and no event left unattributed.
        self.assertTrue(
            all(isinstance(w, int) and 0 <= w <= 1 for w in window_ids),
            f"windowId values must stay within the declared windows, got {window_ids}",
        )
        self.assertNotIn(-1, window_ids, "every event should be attributable")
        # The three events at or inside window 0 belong to it; the boundary event is attributed
        # to the closing window rather than dropped.
        self.assertEqual(window_ids[:4], [0, 0, 0, 0])
        self.assertEqual(window_ids[4:], [1, 1])

    def test_legacy_trace_defaults_to_scripted_provenance(self):
        """Legacy traces without session.provenance must default to 'scripted', never pooled silently."""
        for legacy_version in ("1.1.0", "1.2.0"):
            trace = create_minimal_trace_dict()
            trace["schemaVersion"] = legacy_version
            self.assertNotIn("provenance", trace["session"])

            rows = parse_experiment_trace(trace)
            self.assertEqual(len(rows), 5)
            self.assertTrue(
                all(r["provenance"] == "scripted" for r in rows),
                f"Legacy {legacy_version} rows must default to 'scripted' provenance"
            )

    def test_invalid_provenance_rejected(self):
        """An unrecognised session.provenance value must fail visibly rather than pool silently."""
        trace = create_minimal_trace_dict_v130()
        trace["session"]["provenance"] = "synthetic-ish"
        with self.assertRaises(ValueError) as ctx:
            parse_experiment_trace(trace)
        self.assertIn("Invalid session.provenance", str(ctx.exception))

    def test_unsupported_schema_version_fails_loudly(self):
        """
        An unsupported schemaVersion must raise (never be silently skipped) even when the
        directory also contains ingestible traces, and must name the offending file and version.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_dir = Path(tmp_dir) / "traces"
            trace_dir.mkdir()

            # A supported trace, so a silent skip would still leave the run looking successful.
            supported = create_minimal_trace_dict(session_id="supported-session")
            supported["schemaVersion"] = "1.1.0"
            with open(trace_dir / "supported_110.json", "w", encoding="utf-8") as f:
                json.dump(supported, f)

            # A future, unsupported version.
            future = create_minimal_trace_dict(session_id="future-session")
            future["schemaVersion"] = "1.4.0"
            with open(trace_dir / "future_140.json", "w", encoding="utf-8") as f:
                json.dump(future, f)

            out_p = Path(tmp_dir) / "loud.parquet"
            with self.assertRaises(UnsupportedTraceSchemaError) as ctx:
                ingest_trace_directory(trace_dir, output_path=out_p, force=True)

            message = str(ctx.exception)
            self.assertIn("future_140.json", message)
            self.assertIn("1.4.0", message)
            for supported_version in SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS:
                self.assertIn(supported_version, message)

            # Machine-readable report carried on the exception.
            self.assertEqual(len(ctx.exception.skipped), 1)
            self.assertEqual(Path(ctx.exception.skipped[0]["path"]).name, "future_140.json")
            self.assertEqual(ctx.exception.skipped[0]["schema_version"], "1.4.0")

            self.assertFalse(
                out_p.exists(),
                "A loud failure must not leave a partial canonical Parquet behind"
            )

    def test_lenient_mode_reports_and_skips_unsupported_versions(self):
        """strict=False ingests the supported subset and surfaces the skipped files explicitly."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            trace_dir = Path(tmp_dir) / "traces"
            trace_dir.mkdir()

            supported = create_minimal_trace_dict(session_id="supported-session")
            supported["schemaVersion"] = "1.2.0"
            with open(trace_dir / "supported_120.json", "w", encoding="utf-8") as f:
                json.dump(supported, f)

            future = create_minimal_trace_dict(session_id="future-session")
            future["schemaVersion"] = "1.4.0"
            with open(trace_dir / "future_140.json", "w", encoding="utf-8") as f:
                json.dump(future, f)

            out_p = Path(tmp_dir) / "lenient.parquet"
            result = ingest_trace_directory(trace_dir, output_path=out_p, force=True, strict=False)
            self.assertTrue(os.path.isfile(result))

            df = pq.read_table(result).to_pandas()
            self.assertEqual(set(df["sessionId"].unique()), {"supported-session"})
            self.assertNotIn("future-session", set(df["sessionId"].unique()))

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

        # Classify the live directory by declared version before ingesting.
        declared: dict = {}
        for p in json_traces:
            with open(p, "r", encoding="utf-8") as f:
                declared[p] = json.load(f).get("schemaVersion")

        unsupported = [p for p, v in declared.items() if v not in SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS]
        supported = [p for p, v in declared.items() if v in SUPPORTED_EXPERIMENT_TRACE_SCHEMA_VERSIONS]
        v130 = [p for p, v in declared.items() if v == "1.3.0"]

        def session_ids(paths):
            ids = set()
            for p in paths:
                with open(p, "r", encoding="utf-8") as f:
                    ids.add(json.load(f)["session"]["sessionId"])
            return ids

        supported_sessions = session_ids(supported)
        v130_sessions = session_ids(v130)
        unsupported_sessions = session_ids(unsupported)

        if unsupported:
            # The live directory mixes in obsolete captures: a default run must fail loudly,
            # naming the unsupported files rather than dropping them while reporting success.
            with tempfile.TemporaryDirectory() as tmp_dir:
                strict_out = Path(tmp_dir) / "strict.parquet"
                with self.assertRaises(UnsupportedTraceSchemaError) as ctx:
                    ingest_trace_directory(framework_traces, output_path=strict_out, force=True)
                for p in unsupported:
                    self.assertIn(Path(p).name, str(ctx.exception))
                self.assertFalse(strict_out.exists())

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = Path(tmp_dir) / "real_traces.parquet"
            # The supported subset is ingested only when skipping obsolete versions is explicit.
            result = ingest_trace_directory(framework_traces, output_path=out_p, force=True, strict=False)
            self.assertTrue(os.path.isfile(result))

            table = pq.read_table(result)
            self.assertGreater(table.num_rows, 0)
            df = table.to_pandas()

            # Assert all required columns exist and are non-empty
            for col in ["sessionId", "experimentId", "conditionId", "taskId", "windowId", "sourceEventIds"]:
                self.assertIn(col, df.columns)
                self.assertTrue(df[col].notna().any())

            ingested_sessions = set(df["sessionId"].unique())

            # Every supported trace file, and in particular every declared 1.3.0 file, must be
            # covered by the ingested sessions rather than silently skipped.
            self.assertTrue(
                supported_sessions.issubset(ingested_sessions),
                f"Supported trace sessions missing from ingestion: "
                f"{sorted(supported_sessions - ingested_sessions)}"
            )
            self.assertTrue(
                v130_sessions.issubset(ingested_sessions),
                f"Declared 1.3.0 sessions missing from ingestion: "
                f"{sorted(v130_sessions - ingested_sessions)}"
            )
            if unsupported:
                self.assertFalse(
                    unsupported_sessions & ingested_sessions,
                    "Unsupported-version sessions must not be pooled into the canonical dataset"
                )


if __name__ == "__main__":
    unittest.main()
