"""
test_microtensor_sequence_dataset.py
Unit tests verifying the MicroTensor Sequence Dataset construction (Task 9.2).
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from microtensor_store import (
    MICROTENSOR_SCHEMA,
    SEQUENCE_DATASET_SCHEMA,
    build_microtensor_sequence_dataset,
    load_sequences_from_parquet,
    reconstruct_sequences_from_parquet
)
from training import load_foundation_dataset
from preprocessing import MICROTENSOR_DIM, FEATURE_COLUMN_NAMES, MASK_COLUMN_NAMES, ALL_MICROTENSOR_COLUMNS


def create_synthetic_microtensor_parquet(
    output_path: str,
    sessions_spec: list
) -> str:
    """
    Creates a synthetic MicroTensor Parquet file.
    sessions_spec is a list of tuples: (session_id, task_id, num_windows).
    """
    rows = []
    base_ts = 1000

    for sess_id, task_id, num_wins in sessions_spec:
        for w_idx in range(num_wins):
            w_start = base_ts + w_idx * 250
            w_end = w_start + 500

            row = {
                "dataset_id": "synthetic_testbed",
                "session_id": sess_id,
                "user_id": f"user_{sess_id}",
                "task_id": task_id,
                "window_index": w_idx,
                "window_start_ms": w_start,
                "window_end_ms": w_end,
                "target_label": 1 if w_idx % 2 == 0 else 0,
                "target_name": "CLICK" if w_idx % 2 == 0 else "NO_OUTCOME"
            }
            # Fill 9 features and 9 masks
            for f_idx, col in enumerate(FEATURE_COLUMN_NAMES):
                row[col] = float((w_idx + f_idx) % 10) / 10.0
            for m_idx, col in enumerate(MASK_COLUMN_NAMES):
                row[col] = 1.0

            rows.append(row)

    table = pa.Table.from_pylist(rows, schema=MICROTENSOR_SCHEMA)
    pq.write_table(table, output_path, compression="snappy")
    return output_path


class TestMicroTensorSequenceDataset(unittest.TestCase):
    """Test suite covering Task 9.2 requirements."""

    def test_schema_and_shape_metadata(self):
        """Assert sequence dataset Parquet schema and custom metadata (T=8, D=18, count)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            # 1 session with 10 windows -> produces 10 - 8 + 1 = 3 sequences (T=8)
            create_synthetic_microtensor_parquet(micro_p, [("sess_1", "task_A", 10)])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            self.assertTrue(os.path.isfile(seq_p))
            table = pq.read_table(seq_p)
            self.assertEqual(table.num_rows, 3)

            # Metadata assertions
            meta = table.schema.metadata
            self.assertIsNotNone(meta)
            self.assertEqual(meta.get(b"sequence_length"), b"8")
            self.assertEqual(meta.get(b"feature_dim"), b"18")
            self.assertEqual(meta.get(b"num_examples"), b"3")
            self.assertEqual(meta.get(b"preprocessing_version"), b"1.1.0")
            self.assertEqual(meta.get(b"feature_schema_version"), b"1.1.0")

    def test_zero_session_boundary_leakage(self):
        """Assert no sequence spans across two different sessions."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            # Two sessions with 6 windows each (both < 8 windows)
            # Total windows = 12, but since neither session has >= 8 windows,
            # ZERO sequences should be formed (no cross-session sequences).
            create_synthetic_microtensor_parquet(micro_p, [
                ("sess_1", "task_A", 6),
                ("sess_2", "task_A", 6)
            ])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            table = pq.read_table(seq_p)
            self.assertEqual(table.num_rows, 0, "No sequence must span across sessions")

            # Now give session 1 9 windows and session 2 8 windows
            create_synthetic_microtensor_parquet(micro_p, [
                ("sess_1", "task_A", 9),  # 2 sequences
                ("sess_2", "task_A", 8)   # 1 sequence
            ])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            table = pq.read_table(seq_p)
            self.assertEqual(table.num_rows, 3)

            df = table.to_pandas()
            for _, row in df.iterrows():
                # Verify that all source window IDs are in valid bounds for each session
                if row["session_id"] == "sess_1":
                    self.assertTrue(all(wid < 9 for wid in row["source_window_ids"]))
                elif row["session_id"] == "sess_2":
                    self.assertTrue(all(wid < 8 for wid in row["source_window_ids"]))

    def test_zero_task_boundary_leakage(self):
        """Assert no sequence spans across different tasks within the same session."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            # Single session, but task changes: 5 windows in task_A, 5 windows in task_B
            # Total 10 windows, but each task has only 5 (< 8).
            # ZERO sequences must be constructed.
            create_synthetic_microtensor_parquet(micro_p, [
                ("sess_1", "task_A", 5),
                ("sess_1", "task_B", 5)
            ])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            table = pq.read_table(seq_p)
            self.assertEqual(table.num_rows, 0, "No sequence must span across tasks")

            # In reconstruct_sequences_from_parquet as well:
            X, Y = reconstruct_sequences_from_parquet(micro_p, seq_len=8)
            self.assertEqual(len(X), 0, "reconstruct_sequences_from_parquet must not span tasks")

    def test_lookahead_range_and_anchor_correlation(self):
        """Assert anchor windowId, sourceWindowIds, and lookahead window range metadata."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            create_synthetic_microtensor_parquet(micro_p, [("sess_1", "task_A", 9)])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                lookahead_min_ms=500,
                lookahead_max_ms=1500,
                force=True
            )

            table = pq.read_table(seq_p)
            df = table.to_pandas()

            self.assertEqual(len(df), 2)

            # First sequence: windows 0..7
            row0 = df.iloc[0]
            self.assertEqual(row0["anchor_window_id"], 7)
            self.assertEqual(list(row0["source_window_ids"]), list(range(8)))
            # Lookahead: win_end (1000 + 7*250 + 500 = 3250) + [500, 1500] = [3750, 4750]
            self.assertEqual(row0["window_end_ms"], 3250)
            self.assertEqual(row0["lookahead_start_ms"], 3750)
            self.assertEqual(row0["lookahead_end_ms"], 4750)

            # Second sequence: windows 1..8
            row1 = df.iloc[1]
            self.assertEqual(row1["anchor_window_id"], 8)
            self.assertEqual(list(row1["source_window_ids"]), list(range(1, 9)))
            self.assertEqual(row1["window_end_ms"], 3500)
            self.assertEqual(row1["lookahead_start_ms"], 4000)
            self.assertEqual(row1["lookahead_end_ms"], 5000)

    def test_sequence_tensor_layout_contract(self):
        """Assert sequence tensor layout is identical to foundation training contract (N, 8, 18)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            create_synthetic_microtensor_parquet(micro_p, [("sess_1", "task_A", 12)])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            # Load through load_sequences_from_parquet
            X_seq, Y_seq, meta = load_sequences_from_parquet(seq_p)
            self.assertEqual(X_seq.shape, (5, 8, 18))
            self.assertEqual(X_seq.dtype, np.float32)
            self.assertEqual(Y_seq.shape, (5,))
            self.assertEqual(Y_seq.dtype, np.int64)

            # Load through reconstruct_sequences_from_parquet
            X_rec, Y_rec = reconstruct_sequences_from_parquet(micro_p, seq_len=8)
            self.assertEqual(X_rec.shape, (5, 8, 18))

            # Numerical equivalence between both sequence construction paths
            np.testing.assert_allclose(X_seq, X_rec, atol=1e-6)
            np.testing.assert_array_equal(Y_seq, Y_rec)

    def test_monotonic_window_ordering(self):
        """Assert monotonic window ordering in source_window_ids."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            micro_p = os.path.join(tmp_dir, "test_micro.parquet")
            seq_p = os.path.join(tmp_dir, "test_seq.parquet")

            create_synthetic_microtensor_parquet(micro_p, [("sess_1", "task_A", 10)])

            build_microtensor_sequence_dataset(
                canonical_parquet_path=micro_p,
                output_parquet_path=seq_p,
                seq_len=8,
                force=True
            )

            table = pq.read_table(seq_p)
            df = table.to_pandas()

            for _, row in df.iterrows():
                src_wids = row["source_window_ids"]
                self.assertEqual(len(src_wids), 8)
                self.assertTrue(all(src_wids[i] < src_wids[i + 1] for i in range(len(src_wids) - 1)))
                self.assertEqual(src_wids[-1], row["anchor_window_id"])


if __name__ == "__main__":
    unittest.main()
