"""
test_target_generation.py
Unit tests verifying earliest-event temporal selection, tie-breaking priority,
boundary compliance, session boundedness, OutcomeDataset integration, and class weighting safeguards.
"""

import os
import sys
import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import target_generation
from target_generation import (
    extract_lookahead_outcome,
    extract_outcome_for_window,
    OutcomeDataset,
    compute_class_weights,
    create_sample_outcome_dataset,
    OUTCOME_NAME_TO_ID,
    OUTCOME_TAXONOMY
)


class TestTargetGeneration(unittest.TestCase):
    """Test suite for lookahead outcome extraction and Dataset integration."""

    def test_earliest_event_selection_overrides_priority(self):
        """
        Verify that an earlier outcome (e.g. CLICK at +550ms) takes precedence over
        a later outcome (e.g. FORM_SUBMIT at +1400ms), preserving temporal causality.
        """
        events = [
            {"timestamp_ms": 550, "event_type": "click", "xpath": "/html/body/div"},
            {"timestamp_ms": 1400, "event_type": "click", "xpath": "/html/body/form/input[@type='submit']"}
        ]
        label_id, label_name = extract_lookahead_outcome(events)
        self.assertEqual(label_name, "CLICK")
        self.assertEqual(label_id, OUTCOME_NAME_TO_ID["CLICK"])

    def test_priority_hierarchy_tie_breaking(self):
        """
        Verify that when two outcomes share identical timestamps, the deterministic
        priority hierarchy (FORM_SUBMIT > CLICK > BACKTRACK > RAPID_SCROLL > HOVER_DWELL)
        is applied strictly as a tie-breaker.
        """
        # Tied timestamp at 600ms: generic click vs submit button click
        events = [
            {"timestamp_ms": 600, "event_type": "click", "xpath": "/html/body/div"},
            {"timestamp_ms": 600, "event_type": "click", "xpath": "/html/body/button[@id='submit-btn']"}
        ]
        label_id, label_name = extract_lookahead_outcome(events)
        self.assertEqual(label_name, "FORM_SUBMIT")
        self.assertEqual(label_id, OUTCOME_NAME_TO_ID["FORM_SUBMIT"])

    def test_rapid_scroll_and_hover_dwell_thresholds(self):
        """Verify rapid scroll and hover dwell trigger when their event thresholds are reached."""
        # 4 scroll events
        scroll_evs = [
            {"timestamp_ms": 600, "event_type": "scroll"},
            {"timestamp_ms": 650, "event_type": "scroll"},
            {"timestamp_ms": 700, "event_type": "scroll"},
            {"timestamp_ms": 750, "event_type": "scroll"}
        ]
        label_id, label_name = extract_lookahead_outcome(scroll_evs, rapid_scroll_min_events=4)
        self.assertEqual(label_name, "RAPID_SCROLL")

        # 2 hover events
        hover_evs = [
            {"timestamp_ms": 600, "event_type": "mouseover", "xpath": "/button"},
            {"timestamp_ms": 700, "event_type": "mouseover", "xpath": "/button"}
        ]
        label_id, label_name = extract_lookahead_outcome(hover_evs, hover_dwell_min_events=2)
        self.assertEqual(label_name, "HOVER_DWELL")

    def test_no_outcome_on_inactivity(self):
        """Verify inactivity during ongoing stream produces NO_OUTCOME (not ungrounded affective labels)."""
        empty_evs = []
        label_id, label_name = extract_lookahead_outcome(empty_evs, session_terminated=False)
        self.assertEqual(label_name, "NO_OUTCOME")
        self.assertEqual(label_id, 0)

        # Ambient mouse move only without discrete macro-action
        ambient_evs = [
            {"timestamp_ms": 600, "event_type": "mousemove", "x": 100, "y": 100}
        ]
        label_id, label_name = extract_lookahead_outcome(ambient_evs, session_terminated=False)
        self.assertEqual(label_name, "NO_OUTCOME")
        self.assertEqual(label_id, 0)

    def test_abandon_on_termination_signal(self):
        """Verify explicit termination events (beforeunload, pagehide, unload) or session end produce ABANDON."""
        # Explicit lifecycle event
        unload_evs = [{"timestamp_ms": 800, "event_type": "beforeunload"}]
        label_id, label_name = extract_lookahead_outcome(unload_evs)
        self.assertEqual(label_name, "ABANDON")
        self.assertEqual(label_id, OUTCOME_NAME_TO_ID["ABANDON"])

        # Session termination flag when no events occur
        label_id, label_name = extract_lookahead_outcome([], session_terminated=True)
        self.assertEqual(label_name, "ABANDON")
        self.assertEqual(label_id, OUTCOME_NAME_TO_ID["ABANDON"])

    def test_boundary_conditions_499ms_500ms_1500ms_1501ms(self):
        """
        Verify strict lookahead horizon boundaries:
        - 499ms: Excluded
        - 500ms: Included
        - 1500ms: Included
        - 1501ms: Excluded
        """
        window_end_ms = 1000.0

        # Event at 1499ms (window_end + 499ms) -> Excluded
        evs_499 = [
            {"timestamp_ms": 1499.0, "event_type": "click", "xpath": "/a"},
            {"timestamp_ms": 3000.0, "event_type": "mousemove"}  # keeps session continuing
        ]
        _, name_499 = extract_outcome_for_window(evs_499, window_end_ms)
        self.assertEqual(name_499, "NO_OUTCOME")

        # Event at 1500ms (window_end + 500ms) -> Included
        evs_500 = [{"timestamp_ms": 1500.0, "event_type": "click", "xpath": "/a"}]
        _, name_500 = extract_outcome_for_window(evs_500, window_end_ms)
        self.assertEqual(name_500, "CLICK")

        # Event at 2500ms (window_end + 1500ms) -> Included
        evs_1500 = [{"timestamp_ms": 2500.0, "event_type": "click", "xpath": "/a"}]
        _, name_1500 = extract_outcome_for_window(evs_1500, window_end_ms)
        self.assertEqual(name_1500, "CLICK")

        # Event at 2501ms (window_end + 1501ms) -> Excluded
        evs_1501 = [
            {"timestamp_ms": 2501.0, "event_type": "click", "xpath": "/a"},
            {"timestamp_ms": 3000.0, "event_type": "mousemove"}  # keeps session continuing
        ]
        _, name_1501 = extract_outcome_for_window(evs_1501, window_end_ms)
        self.assertEqual(name_1501, "NO_OUTCOME")

    def test_outcome_dataset_contract_and_dataloader(self):
        """Verify OutcomeDataset shape requirements, tensor types, and DataLoader iteration."""
        X = np.random.uniform(0.0, 1.0, size=(16, 8, 18)).astype(np.float32)
        Y = np.random.randint(0, 7, size=(16,)).astype(np.int64)

        ds = OutcomeDataset(X, Y)
        self.assertEqual(len(ds), 16)

        x0, y0 = ds[0]
        self.assertEqual(x0.shape, (8, 18))
        self.assertEqual(x0.dtype, torch.float32)
        self.assertEqual(y0.dtype, torch.int64)

        loader = DataLoader(ds, batch_size=4, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        self.assertEqual(batch_x.shape, (4, 8, 18))
        self.assertEqual(batch_y.shape, (4,))

    def test_outcome_dataset_shape_validation(self):
        """Ensure OutcomeDataset rejects invalid tensor dimensions."""
        invalid_x = np.random.randn(10, 8, 9)  # 9-D instead of 18-D
        y = np.random.randint(0, 7, size=(10,))
        with self.assertRaises(ValueError):
            OutcomeDataset(invalid_x, y)

    def test_earliest_event_wins_before_priority(self):
        """
        Explicitly verify:
        1. Earliest event timestamp wins regardless of priority (CLICK at 600ms vs FORM_SUBMIT at 900ms -> CLICK).
        2. Priority hierarchy applies strictly as tie-breaker for identical timestamps (600ms vs 600ms -> FORM_SUBMIT).
        """
        # Case 1: Earliest timestamp wins (CLICK @ 600ms vs FORM_SUBMIT @ 900ms)
        evs_diff_time = [
            {"timestamp_ms": 600.0, "event_type": "click", "xpath": "/html/body/div"},
            {"timestamp_ms": 900.0, "event_type": "click", "xpath": "/html/body/form/input[@type='submit']"}
        ]
        label_id1, label_name1 = extract_lookahead_outcome(evs_diff_time)
        self.assertEqual(label_name1, "CLICK")
        self.assertEqual(label_id1, OUTCOME_NAME_TO_ID["CLICK"])

        # Case 2: Tied timestamps resolve by priority (FORM_SUBMIT > CLICK)
        evs_tied_time = [
            {"timestamp_ms": 600.0, "event_type": "click", "xpath": "/html/body/div"},
            {"timestamp_ms": 600.0, "event_type": "click", "xpath": "/html/body/form/input[@type='submit']"}
        ]
        label_id2, label_name2 = extract_lookahead_outcome(evs_tied_time)
        self.assertEqual(label_name2, "FORM_SUBMIT")
        self.assertEqual(label_id2, OUTCOME_NAME_TO_ID["FORM_SUBMIT"])

    def test_observable_abandonment_vs_stream_exhaustion(self):
        """
        Verify:
        - Explicit lifecycle events (beforeunload, pagehide, unload) produce observable abandonment.
        - Stream exhaustion (session_terminated without lifecycle events) produces recording-termination proxy.
        - Invariant: last recorded event != observable abandonment unless explicit lifecycle signal exists.
        """
        # Explicit lifecycle event
        unload_evs = [{"timestamp_ms": 800, "event_type": "beforeunload"}]
        _, name_life, meta_life = extract_lookahead_outcome(unload_evs, return_metadata=True)
        self.assertEqual(name_life, "ABANDON")
        self.assertEqual(meta_life["termination_source"], "lifecycle_event")
        self.assertTrue(meta_life["observable_termination"])

        # Stream exhaustion proxy
        _, name_proxy, meta_proxy = extract_lookahead_outcome([], session_terminated=True, return_metadata=True)
        self.assertEqual(name_proxy, "ABANDON")
        self.assertEqual(meta_proxy["termination_source"], "stream_exhaustion")
        self.assertFalse(meta_proxy["observable_termination"])

        # Continuing session without events -> NO_OUTCOME (not abandonment)
        _, name_cont, meta_cont = extract_lookahead_outcome([], session_terminated=False, return_metadata=True)
        self.assertEqual(name_cont, "NO_OUTCOME")
        self.assertFalse(meta_cont["observable_termination"])

    def test_compute_class_weights_exact_formula(self):
        """
        Verify exact formula w_c = (N + C*alpha) / (C * (N_c + alpha)) without mean normalization.
        """
        # Imbalanced counts: 100 samples total, 7 classes: [40, 30, 10, 10, 5, 3, 2]
        counts = [40, 30, 10, 10, 5, 3, 2]
        targets = []
        for c, cnt in enumerate(counts):
            targets.extend([c] * cnt)
        targets = np.array(targets)
        N = 100.0
        C = 7.0

        # Alpha = 0.0 (exact inverse frequency)
        w_zero = compute_class_weights(targets, num_classes=7, smoothing_alpha=0.0)
        self.assertEqual(w_zero.shape, (7,))
        self.assertEqual(w_zero.dtype, torch.float32)
        for c in range(7):
            expected_w = N / (C * counts[c])
            self.assertAlmostEqual(w_zero[c].item(), expected_w, places=4)

        # Alpha = 2.0 (smoothed inverse frequency)
        alpha = 2.0
        w_smooth = compute_class_weights(targets, num_classes=7, smoothing_alpha=alpha)
        for c in range(7):
            expected_w = (N + C * alpha) / (C * (counts[c] + alpha))
            self.assertAlmostEqual(w_smooth[c].item(), expected_w, places=4)

    def test_compute_class_weights_empty_class_safeguards(self):
        """
        Verify:
        - When allow_empty=False, missing classes raise ValueError EVEN with smoothing_alpha > 0.
        - When allow_empty=True, missing classes receive weight 0.0 (never inflated smoothed weight).
        """
        # Class 4 is missing
        targets_missing = np.array([0]*20 + [1]*10 + [2]*5 + [3]*5 + [5]*20 + [6]*5)

        # allow_empty=False with alpha=0 -> raises ValueError
        with self.assertRaises(ValueError) as ctx1:
            compute_class_weights(targets_missing, num_classes=7, allow_empty=False, smoothing_alpha=0.0)
        self.assertIn("Empty classes detected", str(ctx1.exception))

        # allow_empty=False with alpha=1.0 -> STILL raises ValueError (no silent bypass)
        with self.assertRaises(ValueError) as ctx2:
            compute_class_weights(targets_missing, num_classes=7, allow_empty=False, smoothing_alpha=1.0)
        self.assertIn("Empty classes detected", str(ctx2.exception))

        # allow_empty=True -> missing class 4 gets weight 0.0
        w_allow = compute_class_weights(targets_missing, num_classes=7, allow_empty=True, smoothing_alpha=1.0)
        self.assertEqual(w_allow[4].item(), 0.0)
        self.assertGreater(w_allow[0].item(), 0.0)

    def test_compute_class_weight_diagnostics(self):
        """Verify diagnostic helper reports max, min nonzero, and correct ratio."""
        from target_generation import compute_class_weight_diagnostics
        targets = np.array([0]*50 + [1]*25 + [2]*25)
        diag = compute_class_weight_diagnostics(targets, num_classes=4, smoothing_alpha=0.0, allow_empty=True)

        self.assertIn("max_weight", diag)
        self.assertIn("min_nonzero_weight", diag)
        self.assertIn("max_min_nonzero_ratio", diag)
        self.assertEqual(diag["absent_classes"], [3])
        self.assertAlmostEqual(diag["max_min_nonzero_ratio"], 2.0, places=3)
        self.assertEqual(len(diag["summary_table"]), 4)

    def test_train_only_weighting_invariant(self):
        """
        Demonstrate and test the invariant that class weights computed on the training partition
        must not incorporate validation or test distribution information.
        """
        train_targets = np.array([0]*100 + [1]*50 + [2]*10)
        test_targets = np.array([0]*10 + [1]*50 + [2]*40)  # distribution shift in test

        w_train = compute_class_weights(train_targets, num_classes=3, allow_empty=False)
        w_combined = compute_class_weights(np.concatenate([train_targets, test_targets]), num_classes=3, allow_empty=False)

        # Invariant: Training weights and combined weights differ due to shift; using combined in training leaks test stats
        self.assertFalse(np.allclose(w_train.numpy(), w_combined.numpy()))

    def test_create_sample_outcome_dataset(self):
        """Verify create_sample_outcome_dataset factory produces valid dataset with all 7 classes."""
        ds = create_sample_outcome_dataset(num_samples=35, seq_len=8, from_cache=False)
        self.assertEqual(len(ds), 35)
        x, y = ds[0]
        self.assertEqual(x.shape, (8, 18))
        self.assertEqual(x.shape[-1], 18)

        # Ensure all 7 classes exist in synthetic dataset
        all_labels = set(ds.Y.numpy().tolist())
        self.assertEqual(all_labels, {0, 1, 2, 3, 4, 5, 6})


if __name__ == "__main__":
    unittest.main()
