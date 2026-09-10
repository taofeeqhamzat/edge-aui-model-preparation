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
        """Verify train, validation, and test user sets are strictly disjoint."""
        train_u, val_u, test_u = split_users_leak_free(
            self.micro_p,
            train_ratio=0.60,
            val_ratio=0.20,
            random_seed=42
        )
        self.assertEqual(len(train_u.intersection(val_u)), 0)
        self.assertEqual(len(train_u.intersection(test_u)), 0)
        self.assertEqual(len(val_u.intersection(test_u)), 0)

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
