"""
test_target_dataset_cli.py
In-process coverage for the target-dataset CLI helpers (Task 40.2/40.3 wiring).
"""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import target_dataset
from target_dataset import (
    _print_dataset_summary,
    _resolve_target,
    _split_kwargs,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRACE_DIR = PROJECT_ROOT / ".data" / "raw" / "scripted"


class TestTargetDatasetCli(unittest.TestCase):
    def test_auto_target_follows_the_version(self):
        args = mock.Mock(target="auto", version="v2.0.0-semantic")
        self.assertEqual(_resolve_target(args), "semantic")
        args = mock.Mock(target="auto", version="v1.0.0")
        self.assertEqual(_resolve_target(args), "intervention")
        args = mock.Mock(target="intervention", version="v2.0.0-semantic")
        self.assertEqual(_resolve_target(args), "intervention")

    def test_semantic_split_kwargs_select_the_semantic_label(self):
        args = mock.Mock(
            train_ratio=0.7, val_ratio=0.15, seed=42,
            imbalance_strategy="smoothed", smoothing_alpha=10.0,
        )
        kwargs = _split_kwargs(args, "semantic", {})
        self.assertEqual(kwargs["label_key"], "target_task_outcome")
        self.assertEqual(kwargs["id_key"], "target_task_outcome_id")
        self.assertEqual(len(kwargs["vocabulary"]), 14)
        intervention = _split_kwargs(args, "intervention", {})
        self.assertNotIn("label_key", intervention)

    def test_summary_prints_semantic_and_intervention_scope(self):
        stats = {
            "total_traces": 8, "total_candidate_windows": 10, "total_retained_examples": 5,
            "excluded_missing_context": 1, "target_vocabulary_version": "v",
            "lookahead_horizon_ms": [500, 1500], "excluded_incomplete_lookahead": 2,
            "excluded_recorded_only": 0, "label_sources": {"trace_semantic_outcome": 5},
            "distribution": {
                "counts": {name: 0 for name in target_dataset.INTERVENTION_VOCABULARY},
                "percentages": {name: 0.0 for name in target_dataset.INTERVENTION_VOCABULARY},
                "majority_class": "no_op", "majority_percentage": 0.0,
            },
        }
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_dataset_summary(stats, "semantic")
        self.assertIn("Semantic Label Sources", buffer.getvalue())
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            _print_dataset_summary(stats, "intervention")
        self.assertIn("ADR-013", buffer.getvalue())

    def test_main_writes_a_split_dataset_to_an_explicit_output(self):
        if not TRACE_DIR.is_dir():
            self.skipTest(f"Trace directory not found: {TRACE_DIR}")
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "cli.parquet"
            argv = [
                "target_dataset", "--trace-dir", str(TRACE_DIR),
                "--split", "--output", str(output), "--version", "v2.0.0-semantic",
            ]
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(io.StringIO()):
                main()
            self.assertTrue(output.is_file())

    def test_manifest_and_convenience_copy_helpers(self):
        from target_dataset import (
            _write_convenience_copy,
            _write_manifest,
            build_target_dataset_from_traces,
        )

        if not TRACE_DIR.is_dir():
            self.skipTest(f"Trace directory not found: {TRACE_DIR}")
        rows, _ = build_target_dataset_from_traces(trace_dir=TRACE_DIR)
        with tempfile.TemporaryDirectory() as tmp:
            version_dir = Path(tmp) / "version"
            args = mock.Mock(build_manifest=True, trace_dir=str(TRACE_DIR), version="v-test", seq_len=8)
            _write_manifest(args, rows, None, version_dir)
            self.assertFalse((version_dir / "manifest.json").exists())
            _write_manifest(args, rows, {"class_weights": {"derived_on": "train_partition_only"}}, version_dir)
            self.assertTrue((version_dir / "manifest.json").exists())

            # No convenience copy when an explicit output was requested.
            args = mock.Mock(output=str(Path(tmp) / "out.parquet"), seq_len=8)
            _write_convenience_copy(args, rows, None, Path(tmp) / "out.parquet")

    def test_main_validates_a_manifest(self):
        if not TRACE_DIR.is_dir():
            self.skipTest(f"Trace directory not found: {TRACE_DIR}")
        from data_manager import create_dataset_manifest
        from target_dataset import (
            build_target_dataset_from_traces,
            split_target_dataset,
        )

        with tempfile.TemporaryDirectory() as tmp:
            rows, stats = build_target_dataset_from_traces(
                trace_dir=TRACE_DIR, target="semantic"
            )
            _, summary = split_target_dataset(
                rows,
                vocabulary=list(target_dataset.TARGET_VOCABULARY.values()),
                label_key="target_task_outcome",
                id_key="target_task_outcome_id",
                label_name="semantic",
                build_stats=stats,
            )
            manifest_path = Path(tmp) / "manifest.json"
            create_dataset_manifest(
                rows=rows, split_summary=summary, trace_dir=TRACE_DIR,
                output_path=manifest_path, dataset_version="v2.0.0-test",
            )
            argv = [
                "target_dataset", "--validate-manifest", str(manifest_path),
                "--trace-dir", str(TRACE_DIR),
            ]
            buffer = io.StringIO()
            with mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(buffer):
                main()
            self.assertIn("Manifest validation successful", buffer.getvalue())


if __name__ == "__main__":
    unittest.main()
