"""
test_preprocessing.py
Unit tests for Layer A -> Layer B (Canonical Event Schema)
and Layer B -> Layer C (MicroTensor Extraction with Modality Masking 2D=18).
"""

import os
import sys
import unittest
import numpy as np

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import preprocessing


class TestPreprocessingLayerAB(unittest.TestCase):
    """Task 2.1: Canonical Event Schema & Viewport Normalization."""

    def test_parse_viewport_metadata_valid(self):
        """Parse valid AdSERP XML and verify exact dimensions."""
        _, xml_path = preprocessing.resolve_session_files("p004-b1-t1.csv")
        self.assertIsNotNone(xml_path)
        (vp_w, vp_h), (doc_w, doc_h) = preprocessing.parse_viewport_metadata(xml_path)
        self.assertEqual(vp_w, 1422.0)
        self.assertEqual(vp_h, 1137.0)
        self.assertEqual(doc_w, 1403.0)
        self.assertEqual(doc_h, 2642.0)

    def test_parse_viewport_metadata_missing_file_raises(self):
        """Non-existent XML should raise FileNotFoundError visibly (no synthetic fallback)."""
        with self.assertRaises(FileNotFoundError):
            preprocessing.parse_viewport_metadata("non_existent_file.xml")

    def test_parse_viewport_metadata_malformed_raises(self):
        """Malformed XML missing <window> should raise ValueError visibly."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".xml", mode="w", delete=False) as f:
            f.write("<data><screen>1024x768</screen></data>")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                preprocessing.parse_viewport_metadata(tmp_path)
        finally:
            os.remove(tmp_path)

    def test_compute_window_microtensor_invalid_viewport_raises(self):
        """Non-positive viewport dimensions must raise ValueError visibly."""
        with self.assertRaises(ValueError):
            preprocessing.compute_window_microtensor(
                events=[],
                viewport=(-1.0, 1080.0),
                document=(1920.0, 2000.0)
            )

    def test_parse_adserp_session_bounds_and_types(self):
        """Parse AdSERP session and verify [0, 1] bounds and semantic event types."""
        events = preprocessing.parse_adserp_session("p004-b1-t1.csv")
        self.assertGreater(len(events), 0)

        event_types = set()
        for ev in events:
            # Bounds check
            self.assertGreaterEqual(ev["x_norm"], 0.0)
            self.assertLessEqual(ev["x_norm"], 1.0)
            self.assertGreaterEqual(ev["y_norm"], 0.0)
            self.assertLessEqual(ev["y_norm"], 1.0)

            # Monotonic timestamps
            self.assertIsInstance(ev["timestamp_ms"], int)
            event_types.add(ev["event_type"])

        # Retains semantic event types (mousemove, mouseover, load, etc.)
        self.assertTrue("mousemove" in event_types or "mouseover" in event_types)


class TestMicroTensorExtraction(unittest.TestCase):
    """Task 2.2: MicroTensor Extraction with Modality Masking (2D=18)."""

    def test_mock_microtensor_shape_and_bounds(self):
        """Verify mock tensor shape is (seq_len, 18) and all values in [0, 1]."""
        seq_len = 8
        t = preprocessing.extract_mock_microtensor(seq_len=seq_len, has_scroll=True)
        self.assertEqual(t.shape, (seq_len, 18))
        self.assertFalse(np.isnan(t).any(), "Found NaN in MicroTensor")
        self.assertFalse(np.isinf(t).any(), "Found Inf in MicroTensor")
        self.assertTrue((t >= 0.0).all(), "Values below 0.0")
        self.assertTrue((t <= 1.0).all(), "Values above 1.0")

    def test_modality_mask_behavior(self):
        """Verify binary mask vector M in {0, 1}^9 and concatenated output."""
        # When scroll is disabled
        t_no_scroll = preprocessing.extract_mock_microtensor(seq_len=4, has_scroll=False)
        self.assertEqual(t_no_scroll.shape, (4, 18))
        # Mask is in columns 9..17
        mask = t_no_scroll[0, 9:]
        self.assertEqual(mask.shape, (9,))
        # Core kinematic features active
        self.assertEqual(mask[0], 1.0)
        # Contextual scroll features inactive
        self.assertEqual(mask[7], 0.0)
        self.assertEqual(mask[8], 0.0)
        # When masked with 0, feature values must also be 0
        self.assertEqual(t_no_scroll[0, 7], 0.0)
        self.assertEqual(t_no_scroll[0, 8], 0.0)

    def test_stationary_cursor_with_active_mask(self):
        """Verify stationary cursor produces zero velocity but retains active capability mask (ADR-001)."""
        # Single pointer event or zero movement in window:
        window_evs = [{
            "timestamp_ms": 100,
            "event_type": "mousemove",
            "x_norm": 0.5,
            "y_norm": 0.5,
            "x_raw": 960.0,
            "y_raw": 540.0
        }]
        t = preprocessing.compute_window_microtensor(
            window_evs,
            viewport=(1920.0, 1080.0),
            document=(1920.0, 3000.0),
            has_pointer_support=True,
            has_dom_support=True,
            has_scroll_support=True
        )
        # Kinematics are zero (stationary)
        self.assertEqual(t[0], 0.0)  # meanVelocity
        self.assertEqual(t[1], 0.0)  # maxVelocity
        self.assertEqual(t[2], 0.0)  # meanAcceleration
        self.assertEqual(t[3], 0.0)  # hesitationCount
        self.assertEqual(t[4], 0.0)  # totalTrajectoryLength
        # But pointer modality mask is ACTIVE (sensor is present, observing stillness)
        self.assertEqual(t[9 + 0], 1.0)
        self.assertEqual(t[9 + 1], 1.0)
        self.assertEqual(t[9 + 2], 1.0)
        self.assertEqual(t[9 + 3], 1.0)
        self.assertEqual(t[9 + 4], 1.0)
        self.assertEqual(t[9 + 6], 1.0)

    def test_hover_dwell_with_zero_pointer_motion(self):
        """Verify DOM dwell time accumulates and is preserved even with zero pointer movements."""
        window_evs = [
            {"timestamp_ms": 100, "event_type": "mouseover", "xpath": "//*[@id='submit-btn']"},
            {"timestamp_ms": 200, "event_type": "mouseover", "xpath": "//*[@id='submit-btn']"},
            {"timestamp_ms": 300, "event_type": "mouseover", "xpath": "//*[@id='submit-btn']"}
        ]
        t = preprocessing.compute_window_microtensor(
            window_evs,
            viewport=(1920.0, 1080.0),
            document=(1920.0, 3000.0),
            has_pointer_support=True,
            has_dom_support=True,
            has_scroll_support=True
        )
        # 3 events * 40ms = 120ms / 500ms = 0.24
        self.assertAlmostEqual(t[5], 0.24, places=3)
        # DOM mask is active
        self.assertEqual(t[9 + 5], 1.0)

    def test_sensor_absence_masking(self):
        """Verify missing sensor support deactivates corresponding mask dimensions."""
        window_evs = [
            {"timestamp_ms": 100, "event_type": "mousemove", "x_norm": 0.5, "y_norm": 0.5, "x_raw": 960.0, "y_raw": 540.0},
            {"timestamp_ms": 200, "event_type": "mousemove", "x_norm": 0.6, "y_norm": 0.5, "x_raw": 1152.0, "y_raw": 540.0},
            {"timestamp_ms": 250, "event_type": "scroll"},
            {"timestamp_ms": 300, "event_type": "mouseover", "xpath": "//*[@id='btn']"}
        ]
        # Dataset with pointer only, no DOM and no scroll
        t = preprocessing.compute_window_microtensor(
            window_evs,
            viewport=(1920.0, 1080.0),
            document=(1920.0, 3000.0),
            has_pointer_support=True,
            has_dom_support=False,
            has_scroll_support=False
        )
        # Pointer mask active, DOM and scroll inactive
        self.assertEqual(t[9 + 0], 1.0)
        self.assertEqual(t[9 + 5], 0.0)  # DOM mask
        self.assertEqual(t[9 + 7], 0.0)  # scroll depth mask
        self.assertEqual(t[9 + 8], 0.0)  # scroll vel mask
        # Masked features must be strictly 0.0
        self.assertEqual(t[5], 0.0)
        self.assertEqual(t[7], 0.0)
        self.assertEqual(t[8], 0.0)


    def test_session_extraction_on_real_adserp(self):
        """Test extraction directly on real AdSERP session events."""
        events = preprocessing.parse_adserp_session("p004-b1-t1.csv")
        t_seq = preprocessing.extract_session_microtensors(events, window_size_ms=500, stride_ms=250)
        self.assertGreater(len(t_seq), 0)
        self.assertEqual(t_seq.shape[-1], 18)
        self.assertFalse(np.isnan(t_seq).any())
        self.assertFalse(np.isinf(t_seq).any())
        self.assertTrue((t_seq >= 0.0).all())
        self.assertTrue((t_seq <= 1.0).all())


if __name__ == "__main__":
    unittest.main()
