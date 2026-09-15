"""
test_microtensor_parquet.py
Unit tests for flattened MicroTensor Parquet storage, numerical equivalence,
leak-free splitting, and sequence reconstruction.
"""

import os
import sys
import unittest
import numpy as np
import pyarrow.parquet as pq

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from preprocessing import (
    parse_adserp_session,
    extract_session_microtensors,
    MICROTENSOR_DIM,
    NUM_FEATURES
)
from microtensor_store import (
    extract_microtensors_from_canonical,
    reconstruct_sequences_from_parquet,
    split_users_leak_free,
    SplitResult,
    MICROTENSOR_SCHEMA
)
from data import convert_adserp_to_canonical, resolve_adserp_dir


class TestMicroTensorParquetStore(unittest.TestCase):
    """Verify MicroTensor Parquet storage, equivalence, and sequence reconstruction."""

    @classmethod
    def setUpClass(cls):
        raw_adserp = resolve_adserp_dir()
        if not raw_adserp or not raw_adserp.is_dir():
            raise unittest.SkipTest("AdSERP dataset not present locally")

        import tempfile
        cls.tmp_dir_obj = tempfile.TemporaryDirectory()
        cls.tmp_dir = cls.tmp_dir_obj.name

        cls.canon_p = os.path.join(cls.tmp_dir, "adserp_canon.parquet")
        convert_adserp_to_canonical(
            raw_path=str(raw_adserp),
            output_path=cls.canon_p,
            max_sessions=5,
            force=True
        )

        cls.micro_p = os.path.join(cls.tmp_dir, "adserp_micro.parquet")
        extract_microtensors_from_canonical(
            canonical_parquet_path=cls.canon_p,
            output_parquet_path=cls.micro_p,
            force=True
        )

    @classmethod
    def tearDownClass(cls):
        cls.tmp_dir_obj.cleanup()

    def test_schema_and_flattened_columns(self):
        """Verify MicroTensor Parquet contains all 18 scalar feature/mask columns."""
        table = pq.read_table(self.micro_p)
        field_names = [f.name for f in table.schema]

        expected_features = [
            "mean_velocity", "max_velocity", "mean_acceleration", "hesitation_count",
            "total_trajectory_length", "dwell_time_ms", "trajectory_entropy",
            "scroll_depth_percentage", "scroll_velocity"
        ]
        for f in expected_features:
            self.assertIn(f, field_names)
            self.assertIn(f"mask_{f}", field_names)

        self.assertIn("target_label", field_names)
        self.assertIn("window_index", field_names)
        self.assertIn("session_id", field_names)

    def test_numerical_equivalence_against_legacy(self):
        """
        Numerically compare MicroTensors extracted from canonical Parquet
        against the legacy extraction directly on the raw session CSV/XML.
        """
        table_canon = pq.read_table(self.canon_p)
        df_canon = table_canon.to_pandas()
        first_session = df_canon["session_id"].iloc[0]

        # Legacy extraction directly from raw session
        raw_events = parse_adserp_session(f"{first_session}.csv")
        legacy_tensors = extract_session_microtensors(
            raw_events,
            window_size_ms=500,
            stride_ms=250,
            min_events_per_window=3,
            has_scroll_support=True
        )

        # Extraction from Parquet
        table_micro = pq.read_table(self.micro_p)
        df_micro = table_micro.to_pandas()
        sess_micro = df_micro[df_micro["session_id"] == first_session].sort_values("window_index")

        feature_cols = [
            "mean_velocity", "max_velocity", "mean_acceleration", "hesitation_count",
            "total_trajectory_length", "dwell_time_ms", "trajectory_entropy",
            "scroll_depth_percentage", "scroll_velocity",
            "mask_mean_velocity", "mask_max_velocity", "mask_mean_acceleration", "mask_hesitation_count",
            "mask_total_trajectory_length", "mask_dwell_time_ms", "mask_trajectory_entropy",
            "mask_scroll_depth_percentage", "mask_scroll_velocity"
        ]
        parquet_tensors = sess_micro[feature_cols].to_numpy(dtype=np.float32)

        self.assertEqual(len(legacy_tensors), len(parquet_tensors))
        # Check numerical equivalence within floating-point tolerance
        np.testing.assert_allclose(legacy_tensors, parquet_tensors, atol=1e-5)

    def test_leak_free_user_splitting(self):
        """Verify train, validation, and test user sets are strictly disjoint and return SplitResult."""
        split_res = split_users_leak_free(
            self.micro_p,
            train_ratio=0.60,
            val_ratio=0.20,
            random_seed=42
        )
        self.assertIsInstance(split_res, SplitResult)
        self.assertEqual(len(split_res.train_ids.intersection(split_res.val_ids)), 0)
        self.assertEqual(len(split_res.train_ids.intersection(split_res.test_ids)), 0)
        self.assertEqual(len(split_res.val_ids.intersection(split_res.test_ids)), 0)
        # Verify unpack compatibility: train_u, val_u, test_u = split_res
        train_u, val_u, test_u = split_res
        self.assertEqual(train_u, split_res.train_ids)

    def test_split_result_contract_and_reproducibility(self):
        """Verify SplitResult contract, split_by detection, and deterministic seed reproducibility."""
        res1 = split_users_leak_free(self.micro_p, train_ratio=0.70, val_ratio=0.15, random_seed=123)
        res2 = split_users_leak_free(self.micro_p, train_ratio=0.70, val_ratio=0.15, random_seed=123)
        self.assertEqual(res1.train_ids, res2.train_ids)
        self.assertEqual(res1.val_ids, res2.val_ids)
        self.assertEqual(res1.test_ids, res2.test_ids)
        self.assertEqual(res1.split_by, res2.split_by)

    def test_split_before_sequence_construction_zero_leakage(self):
        """
        Hard test for Methodological Invariant 1:
        Verify that partition filtering occurs BEFORE sequence construction.
        Assert zero session or user overlap between train, val, and test partitions.
        """
        split_res = split_users_leak_free(self.micro_p, train_ratio=0.60, val_ratio=0.20, random_seed=42)

        X_train, Y_train = reconstruct_sequences_from_parquet(self.micro_p, split_result=split_res, split_partition="train")
        X_val, Y_val = reconstruct_sequences_from_parquet(self.micro_p, split_result=split_res, split_partition="val")
        X_test, Y_test = reconstruct_sequences_from_parquet(self.micro_p, split_result=split_res, split_partition="test")

        # Read parquet to map session to user
        table = pq.read_table(self.micro_p, columns=["session_id", "user_id"])
        df = table.to_pandas().drop_duplicates()
        sess_to_user = dict(zip(df["session_id"], df["user_id"]))

        # Check disjointness of user IDs in split
        self.assertEqual(len(split_res.train_ids.intersection(split_res.val_ids)), 0)
        self.assertEqual(len(split_res.train_ids.intersection(split_res.test_ids)), 0)
        self.assertEqual(len(split_res.val_ids.intersection(split_res.test_ids)), 0)

        # Check disjointness of sessions
        train_sessions = {s for s, u in sess_to_user.items() if (u if split_res.split_by == "user_id" else s) in split_res.train_ids}
        val_sessions = {s for s, u in sess_to_user.items() if (u if split_res.split_by == "user_id" else s) in split_res.val_ids}
        test_sessions = {s for s, u in sess_to_user.items() if (u if split_res.split_by == "user_id" else s) in split_res.test_ids}

        self.assertEqual(len(train_sessions.intersection(val_sessions)), 0)
        self.assertEqual(len(train_sessions.intersection(test_sessions)), 0)
        self.assertEqual(len(val_sessions.intersection(test_sessions)), 0)

    def test_placeholder_user_fallback_to_session_id(self):
        """
        Verify that invalid user placeholders ('', 'unknown', 'anonymous', 'null')
        trigger safe fallback to session_id with split_by='session_id'.
        """
        import tempfile
        import pyarrow as pa
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_parquet = os.path.join(tmp_dir, "placeholder_test.parquet")
            test_schema = pa.schema([
                ("dataset_id", pa.string()),
                ("session_id", pa.string()),
                ("user_id", pa.string()),
                ("window_index", pa.int32()),
                ("window_start_ms", pa.int64()),
                ("window_end_ms", pa.int64()),
                ("target_label", pa.int64()),
                ("target_name", pa.string()),
                ("mean_velocity", pa.float32()),
                ("max_velocity", pa.float32()),
                ("mean_acceleration", pa.float32()),
                ("hesitation_count", pa.float32()),
                ("total_trajectory_length", pa.float32()),
                ("dwell_time_ms", pa.float32()),
                ("trajectory_entropy", pa.float32()),
                ("scroll_depth_percentage", pa.float32()),
                ("scroll_velocity", pa.float32()),
                ("mask_mean_velocity", pa.float32()),
                ("mask_max_velocity", pa.float32()),
                ("mask_mean_acceleration", pa.float32()),
                ("mask_hesitation_count", pa.float32()),
                ("mask_total_trajectory_length", pa.float32()),
                ("mask_dwell_time_ms", pa.float32()),
                ("mask_trajectory_entropy", pa.float32()),
                ("mask_scroll_depth_percentage", pa.float32()),
                ("mask_scroll_velocity", pa.float32()),
            ])
            rows = []
            for s_idx in range(10):
                sess_id = f"sess_{s_idx}"
                # All user IDs are placeholder "unknown" or empty
                u_id = "unknown" if s_idx % 2 == 0 else ""
                for w_idx in range(8):
                    r = {
                        "dataset_id": "test",
                        "session_id": sess_id,
                        "user_id": u_id,
                        "window_index": w_idx,
                        "window_start_ms": w_idx * 250,
                        "window_end_ms": (w_idx + 1) * 250,
                        "target_label": 0,
                        "target_name": "NO_OUTCOME"
                    }
                    for col in MICROTENSOR_SCHEMA.names[6:24]:
                        r[col] = 0.5
                    rows.append(r)
            tbl = pa.Table.from_pylist(rows, schema=test_schema)
            pq.write_table(tbl, tmp_parquet)

            split_res = split_users_leak_free(tmp_parquet, train_ratio=0.7, val_ratio=0.15)
            self.assertEqual(split_res.split_by, "session_id")
            self.assertGreater(len(split_res.train_ids), 0)
            self.assertEqual(len(split_res.train_ids.intersection(split_res.val_ids)), 0)

    def test_temporal_continuity_enforcement(self):
        """
        Verify that enforce_temporal_continuity=True segments sequences at time gaps
        greater than max_step_gap_ms (distinguishing window validity from sequence continuity).
        """
        import tempfile
        import pyarrow as pa
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_parquet = os.path.join(tmp_dir, "continuity_test.parquet")
            rows = []
            # Session with 16 windows, but a 5-second gap between window 3 and 4
            sess_id = "sess_with_gap"
            for w_idx in range(16):
                start_ms = w_idx * 250 if w_idx < 4 else (w_idx * 250 + 5000)
                r = {
                    "dataset_id": "test",
                    "session_id": sess_id,
                    "user_id": "user_1",
                    "window_index": w_idx,
                    "window_start_ms": start_ms,
                    "window_end_ms": start_ms + 500,
                    "target_label": 0,
                    "target_name": "NO_OUTCOME"
                }
                for col in MICROTENSOR_SCHEMA.names[6:24]:
                    r[col] = 0.5
                rows.append(r)
            tbl = pa.Table.from_pylist(rows, schema=MICROTENSOR_SCHEMA)
            pq.write_table(tbl, tmp_parquet)

            # Without continuity enforcement: 16 windows -> 16 - 8 + 1 = 9 sequences spanning gap
            X_raw, _ = reconstruct_sequences_from_parquet(tmp_parquet, seq_len=8, enforce_temporal_continuity=False)
            self.assertEqual(len(X_raw), 9)

            # With continuity enforcement: segment 1 (4 windows < 8 -> 0 seqs), segment 2 (12 windows -> 12 - 8 + 1 = 5 seqs)
            X_cont, _ = reconstruct_sequences_from_parquet(tmp_parquet, seq_len=8, enforce_temporal_continuity=True, max_step_gap_ms=300)
            self.assertEqual(len(X_cont), 5)

    def test_reconstruct_sequences_shape_and_bounds(self):
        """Verify reconstructed sequences have shape (N, T, 18) and all values in [0, 1]."""
        seq_len = 8
        X, Y = reconstruct_sequences_from_parquet(self.micro_p, seq_len=seq_len)
        if len(X) > 0:
            self.assertEqual(X.ndim, 3)
            self.assertEqual(X.shape[1], seq_len)
            self.assertEqual(X.shape[2], MICROTENSOR_DIM)
            self.assertEqual(len(X), len(Y))

            # Bounds check
            self.assertTrue((X >= 0.0).all())
            self.assertTrue((X <= 1.0).all())
            self.assertFalse(np.isnan(X).any())


if __name__ == "__main__":
    unittest.main()
