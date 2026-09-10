"""
test_canonical_schema.py
Unit tests verifying the Cross-Dataset Canonical Event Schema and Dataset Adapters.
"""

import os
import sys
import unittest
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from data import (
    CANONICAL_EVENT_SCHEMA,
    convert_adserp_to_canonical,
    convert_continuous_kinematics_to_canonical,
    convert_high_volume_trajectories_to_canonical,
    resolve_adserp_dir,
    resolve_continuous_kinematics_dir,
    resolve_high_volume_trajectories_dir
)


class TestCanonicalSchema(unittest.TestCase):
    """Verify Canonical Event Schema and Adapter contracts."""

    def test_schema_field_definitions(self):
        """Verify CANONICAL_EVENT_SCHEMA contains all required standard fields."""
        self.assertIsNotNone(CANONICAL_EVENT_SCHEMA)
        field_names = [f.name for f in CANONICAL_EVENT_SCHEMA]
        
        required_fields = [
            "dataset_id", "session_id", "user_id", "trial_id",
            "timestamp_ms", "event_type", "x", "y", "x_norm", "y_norm",
            "viewport_width", "viewport_height", "document_width", "document_height",
            "target_id", "source_event_id", "duration_ms", "angle_deg", "distance_px", "velocity_px_s"
        ]
        for req in required_fields:
            self.assertIn(req, field_names, f"Missing required field {req} in CANONICAL_EVENT_SCHEMA")

        # Type checks
        self.assertEqual(CANONICAL_EVENT_SCHEMA.field("timestamp_ms").type, pa.int64())
        self.assertEqual(CANONICAL_EVENT_SCHEMA.field("x_norm").type, pa.float32())
        self.assertEqual(CANONICAL_EVENT_SCHEMA.field("y_norm").type, pa.float32())

    def test_adserp_canonical_conversion(self):
        """Convert a small sample of AdSERP sessions and verify canonical row integrity."""
        raw_adserp = resolve_adserp_dir()
        if not raw_adserp or not raw_adserp.is_dir():
            self.skipTest("AdSERP dataset not present locally")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "adserp_test.parquet")
            result_path = convert_adserp_to_canonical(
                raw_path=str(raw_adserp),
                output_path=out_p,
                max_sessions=3,
                force=True
            )
            self.assertTrue(os.path.isfile(result_path))

            table = pq.read_table(result_path)
            self.assertGreater(table.num_rows, 0)
            df = table.to_pandas()

            # Verify dataset_id
            self.assertTrue((df["dataset_id"] == "adserp").all())

            # Verify user_id prefix (pXXX)
            for u in df["user_id"].dropna().unique():
                self.assertTrue(str(u).startswith("p"), f"User ID {u} does not match pXXX format")

            # Verify normalized bounds
            x_norm = df["x_norm"].dropna()
            y_norm = df["y_norm"].dropna()
            self.assertTrue((x_norm >= 0.0).all() and (x_norm <= 1.0).all())
            self.assertTrue((y_norm >= 0.0).all() and (y_norm <= 1.0).all())

            # Verify viewport dimensions are positive
            self.assertTrue((df["viewport_width"] > 0).all())
            self.assertTrue((df["viewport_height"] > 0).all())

    def test_continuous_kinematics_canonical_conversion(self):
        """Convert a small sample of Continuous Kinematics and verify participant joining."""
        raw_ck = resolve_continuous_kinematics_dir()
        if not raw_ck or not raw_ck.is_dir():
            self.skipTest("Continuous Kinematics dataset not present locally")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "ck_test.parquet")
            result_path = convert_continuous_kinematics_to_canonical(
                raw_path=str(raw_ck),
                output_path=out_p,
                max_sessions=3,
                force=True
            )
            self.assertTrue(os.path.isfile(result_path))

            table = pq.read_table(result_path)
            self.assertGreater(table.num_rows, 0)
            df = table.to_pandas()

            self.assertTrue((df["dataset_id"] == "continuous_kinematics").all())
            # Verify user_id was mapped from participants.tsv
            self.assertTrue(df["user_id"].notna().any(), "user_id was not populated from participants.tsv")

    def test_high_volume_trajectories_non_fabrication(self):
        """Convert a small sample of HVT and verify coordinates/viewports are strictly NULL."""
        raw_hvt = resolve_high_volume_trajectories_dir()
        if not raw_hvt or not raw_hvt.is_dir():
            self.skipTest("High-Volume Trajectories dataset not present locally")

        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_p = os.path.join(tmp_dir, "hvt_test.parquet")
            result_path = convert_high_volume_trajectories_to_canonical(
                raw_path=str(raw_hvt),
                output_path=out_p,
                max_rows=100,
                force=True
            )
            self.assertTrue(os.path.isfile(result_path))

            table = pq.read_table(result_path)
            self.assertEqual(table.num_rows, 100)
            df = table.to_pandas()

            self.assertTrue((df["dataset_id"] == "high_volume_trajectories").all())
            # Non-fabrication: x, y, viewport must be strictly NULL
            self.assertTrue(df["x"].isna().all(), "Fabricated x coordinate detected in HVT")
            self.assertTrue(df["y"].isna().all(), "Fabricated y coordinate detected in HVT")
            self.assertTrue(df["viewport_width"].isna().all(), "Fabricated viewport width detected in HVT")
            self.assertTrue(df["viewport_height"].isna().all(), "Fabricated viewport height detected in HVT")

            # Kinematics must be present
            self.assertTrue(df["angle_deg"].notna().any() or df["duration_ms"].notna().any())


if __name__ == "__main__":
    unittest.main()
