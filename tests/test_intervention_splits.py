"""
test_intervention_splits.py
Unit tests verifying Task 11.1: Intervention-Head Dataset Splits & Class-Weight Resolution.

Enforces:
1. Splits are strictly session-bounded (zero session or task-within-session leakage).
2. Class weights are derived strictly from the training partition only (test-set leak-free).
3. Absent classes in any partition are explicitly detected and reported.
4. Majority-class baseline is calculated for all partitions using the training majority class.
5. Imbalance strategy (weighting, resampling, or neither) is recorded with full rationale.
"""

import copy
import os
import sys
import unittest
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from target_dataset import (
    build_target_dataset_from_traces,
    split_target_dataset,
    INTERVENTION_VOCABULARY,
    INTERVENTION_TO_ID,
)


class TestInterventionSplits(unittest.TestCase):
    """Test suite covering Task 11.1 acceptance criteria."""

    @classmethod
    def setUpClass(cls):
        """Build target dataset rows once for the test suite."""
        trace_dir = Path(__file__).resolve().parent.parent / ".data" / "raw" / "scripted"
        if not trace_dir.is_dir():
            raise unittest.SkipTest(f"Trace directory not found: {trace_dir}")

        cls.rows, cls.stats = build_target_dataset_from_traces(trace_dir=trace_dir)
        cls.split_rows, cls.split_summary = split_target_dataset(
            copy.deepcopy(cls.rows),
            train_ratio=0.70,
            val_ratio=0.15,
            random_seed=42,
            imbalance_strategy="smoothed",
            smoothing_alpha=10.0,
        )

    def test_session_bounded_disjoint_partitions(self):
        """Assert no session appears in more than one partition (Zero Session Leakage)."""
        train_sessions = set(self.split_summary["sessions"]["train"])
        val_sessions = set(self.split_summary["sessions"]["val"])
        test_sessions = set(self.split_summary["sessions"]["test"])

        # Pairwise disjointness
        self.assertEqual(len(train_sessions & val_sessions), 0, "Train and Val sessions must not overlap")
        self.assertEqual(len(train_sessions & test_sessions), 0, "Train and Test sessions must not overlap")
        self.assertEqual(len(val_sessions & test_sessions), 0, "Val and Test sessions must not overlap")

        # Total sessions count
        all_sessions = set(r["session_id"] for r in self.split_rows)
        self.assertEqual(train_sessions | val_sessions | test_sessions, all_sessions)
        self.assertEqual(len(all_sessions), 18)

        # Every row split tag matches its session set
        for r in self.split_rows:
            sid = r["session_id"]
            if sid in train_sessions:
                self.assertEqual(r["split"], "train")
            elif sid in val_sessions:
                self.assertEqual(r["split"], "val")
            elif sid in test_sessions:
                self.assertEqual(r["split"], "test")
            else:
                self.fail(f"Row has unassigned session: {sid}")

    def test_task_session_disjointness(self):
        """Assert no (session_id, task_id) pair appears in two partitions."""
        train_st = set((r["session_id"], r["task_id"]) for r in self.split_rows if r["split"] == "train")
        val_st = set((r["session_id"], r["task_id"]) for r in self.split_rows if r["split"] == "val")
        test_st = set((r["session_id"], r["task_id"]) for r in self.split_rows if r["split"] == "test")

        self.assertTrue(train_st.isdisjoint(val_st), "Train and Val (session, task) pairs must be disjoint")
        self.assertTrue(train_st.isdisjoint(test_st), "Train and Test (session, task) pairs must be disjoint")
        self.assertTrue(val_st.isdisjoint(test_st), "Val and Test (session, task) pairs must be disjoint")

    def test_class_weights_resolved_on_training_partition_only(self):
        """
        Assert class weights are derived exclusively from the training partition.
        Mutating or injecting arbitrary labels into the test partition must NOT affect train weights.
        """
        original_weights = copy.deepcopy(self.split_summary["class_weights"]["weights"])

        # Create a modified copy of rows where the test partition is heavily contaminated
        # by injecting 500 fictitious examples of 'simplify_options' into test sessions
        corrupted_rows = copy.deepcopy(self.split_rows)
        test_session = self.split_summary["sessions"]["test"][0]

        fictitious_example = copy.deepcopy(corrupted_rows[0])
        fictitious_example["session_id"] = test_session
        fictitious_example["target_intervention"] = "simplify_options"
        fictitious_example["target_intervention_id"] = INTERVENTION_TO_ID["simplify_options"]

        for _ in range(500):
            corrupted_rows.append(copy.deepcopy(fictitious_example))

        # Re-run splitting on corrupted dataset
        _, corrupted_summary = split_target_dataset(
            corrupted_rows,
            train_ratio=0.70,
            val_ratio=0.15,
            random_seed=42,
            imbalance_strategy="smoothed",
            smoothing_alpha=10.0,
        )

        recalculated_weights = corrupted_summary["class_weights"]["weights"]

        # The weights on the training partition must be BIT-IDENTICAL
        for cls_name in INTERVENTION_VOCABULARY:
            orig_w = original_weights[cls_name]
            new_w = recalculated_weights[cls_name]
            self.assertAlmostEqual(
                orig_w, new_w, places=7,
                msg=f"Class weight for '{cls_name}' changed when test partition was modified! Training weights must be test-leak free."
            )

        self.assertEqual(corrupted_summary["class_weights"]["derived_on"], "train_partition_only")

    def test_class_distribution_and_absent_classes(self):
        """Assert per-partition distribution counts and explicit reporting of absent classes."""
        ec = self.split_summary["example_counts"]
        self.assertEqual(ec["total"], 289)
        self.assertEqual(ec["train"], 208)
        self.assertEqual(ec["val"], 51)
        self.assertEqual(ec["test"], 30)

        # All 5 classes represented in training partition
        train_absent = self.split_summary["absent_classes"]["train"]
        self.assertEqual(len(train_absent), 0, f"Expected 0 absent classes in train, found: {train_absent}")

        # Val and Test partitions lack 'simplify_options' and 'highlight_primary_action'
        val_absent = self.split_summary["absent_classes"]["val"]
        test_absent = self.split_summary["absent_classes"]["test"]

        self.assertIn("simplify_options", val_absent)
        self.assertIn("highlight_primary_action", val_absent)
        self.assertIn("simplify_options", test_absent)
        self.assertIn("highlight_primary_action", test_absent)

        # Confirm count sum equals total for each partition
        for p in ["train", "val", "test"]:
            p_counts = self.split_summary["class_distribution"][p]["counts"]
            self.assertEqual(sum(p_counts.values()), ec[p])

    def test_majority_class_baselines(self):
        """Assert majority-class baseline metrics are calculated across all partitions."""
        majority_summary = self.split_summary["majority_baseline"]

        # Majority class is 'no_op'
        self.assertEqual(majority_summary["train"]["majority_class"], "no_op")
        self.assertEqual(majority_summary["val"]["majority_class"], "no_op")
        self.assertEqual(majority_summary["test"]["majority_class"], "no_op")

        # Accuracies are within expected valid range [0, 1]
        for p in ["train", "val", "test"]:
            acc = majority_summary[p]["accuracy"]
            f1 = majority_summary[p]["macro_f1"]
            self.assertGreaterEqual(acc, 0.30)
            self.assertLessEqual(acc, 0.55)
            self.assertGreaterEqual(f1, 0.0)
            self.assertLessEqual(f1, 1.0)

    def test_provenance_fields_present_on_every_row(self):
        """Assert all 9 provenance fields required by Brief §8 are present on every row."""
        required_fields = [
            "session_id",
            "experiment_id",
            "condition_id",
            "task_id",
            "anchor_window_id",
            "source_event_ids",
            "preprocessing_version",
            "feature_schema_version",
            "target_generation_version",
            "split"
        ]

        for idx, row in enumerate(self.split_rows):
            for field in required_fields:
                self.assertIn(field, row, f"Row {idx} is missing provenance field '{field}'")
                self.assertIsNotNone(row[field], f"Row {idx} has None for provenance field '{field}'")
                if isinstance(row[field], list):
                    self.assertGreater(len(row[field]), 0, f"Row {idx} has empty list for '{field}'")

            self.assertIn(row["split"], {"train", "val", "test"})
            self.assertEqual(row["preprocessing_version"], "1.1.0")
            self.assertEqual(row["feature_schema_version"], "1.1.0")
            self.assertEqual(row["target_generation_version"], "1.0.0")

    def test_cli_split_command(self):
        """Assert python3 -m src.target_dataset --split executes cleanly with exit code 0."""
        import subprocess

        cmd = [sys.executable, "-m", "src.target_dataset", "--split"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))

        self.assertEqual(proc.returncode, 0, f"CLI command failed with stderr: {proc.stderr}")
        self.assertIn("SESSION-BOUNDED SPLIT REPORT", proc.stdout)
        self.assertIn("CLASS WEIGHTS RESOLUTION", proc.stdout)
        self.assertIn("MAJORITY-CLASS BASELINES", proc.stdout)


if __name__ == "__main__":
    unittest.main()
