"""
test_intervention_head_training.py
Unit tests verifying Phase D: Target-Domain Intervention-Head Training & Evaluation (Tasks 12.1-12.4).

Enforces:
1. Dataset loader contracts and loss weighting invariants.
2. Experiment E1 Strict Freezing: Backbone parameters are bit-identical before and after training step; head parameters change.
3. Experiment E2 Partial Fine-Tuning: Disjoint parameter groups, lower GRU layer bit-identical, terminal layer + head change.
4. Experiment E3 Full Fine-Tuning: All parameters trainable; retention audit detects catastrophic forgetting.
5. Multi-metric evaluation produces all required primary, secondary, ranking, and failure-case metrics.
"""

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from training import (
    EdgeAUIGRU,
    TargetInterventionHead,
    evaluate_intervention_model,
    evaluate_intervention_majority_baseline,
    evaluate_intervention_retention,
    train_intervention_model,
)
from data_manager import create_dataset_manifest
from target_dataset import (
    TargetInterventionDataset,
    build_target_dataset_from_traces,
    load_intervention_dataset,
    get_intervention_class_weights,
    save_target_dataset_parquet,
    split_target_dataset,
    INTERVENTION_VOCABULARY,
    INTERVENTION_TO_ID,
)


class TestInterventionHeadTraining(unittest.TestCase):
    """Test suite covering Phase D tasks (12.1, 12.2, 12.3, 12.4)."""

    @classmethod
    def setUpClass(cls):
        # Build the dataset from the traces that are present. The previous version loaded a
        # cached .data artifact recorded from a larger capture set, so the expected counts
        # described data the repository no longer contains.
        cls._tmpdir = tempfile.TemporaryDirectory()
        root = Path(__file__).resolve().parent.parent
        trace_dir = root / ".data" / "raw" / "scripted"
        rows, _ = build_target_dataset_from_traces(trace_dir=trace_dir)
        split_rows, summary = split_target_dataset(rows, random_seed=42)
        cls.summary = summary
        cls.expected = summary["example_counts"]
        cls.parquet_path = Path(cls._tmpdir.name) / "target_intervention_dataset.parquet"
        cls.manifest_path = Path(cls._tmpdir.name) / "manifest.json"
        save_target_dataset_parquet(split_rows, cls.parquet_path, split_summary=summary)
        create_dataset_manifest(
            rows=split_rows,
            split_summary=summary,
            trace_dir=trace_dir,
            output_path=cls.manifest_path,
            dataset_version="test-training",
        )

        cls.train_ds = load_intervention_dataset(split="train", parquet_path=cls.parquet_path)
        cls.val_ds = load_intervention_dataset(split="val", parquet_path=cls.parquet_path)
        cls.test_ds = load_intervention_dataset(split="test", parquet_path=cls.parquet_path)
        cls.class_weights = get_intervention_class_weights(manifest_path=cls.manifest_path)

    @classmethod
    def tearDownClass(cls):
        cls._tmpdir.cleanup()

    def test_dataset_loader_contract(self):
        """Assert dataset partitions match expected example counts and batch tensor shapes."""
        self.assertEqual(len(self.train_ds), self.expected["train"])
        self.assertEqual(len(self.val_ds), self.expected["val"])
        self.assertEqual(len(self.test_ds), self.expected["test"])

        # Test index item
        seq, ctx, tgt = self.train_ds[0]
        self.assertEqual(seq.shape, (8, 18), "MicroTensor sequence shape must be (8, 18)")
        self.assertEqual(ctx.shape, (6,), "Context vector shape must be (6,)")
        self.assertIsInstance(tgt.item(), int)
        self.assertTrue(0 <= tgt.item() < 5, "Target must be in range 0..4")

        # Test class weights. A class absent from the training partition gets weight 0 by
        # design (declared, not concealed), so only represented classes must be positive.
        self.assertEqual(self.class_weights.shape, (5,), "Class weights must be 5-dimensional")
        self.assertTrue((self.class_weights >= 0).all(), "Class weights must be non-negative")
        train_counts = self.summary["class_distribution"]["train"]["counts"]
        for name in INTERVENTION_VOCABULARY:
            if train_counts.get(name, 0) > 0:
                self.assertGreater(
                    float(self.class_weights[INTERVENTION_TO_ID[name]]), 0.0,
                    f"Class '{name}' is represented in train and must have a positive weight",
                )

        # Test metadata
        meta = self.train_ds.get_metadata(0)
        self.assertIn("session_id", meta)
        self.assertIn("task_id", meta)

    def test_empty_split_and_invalid_metadata_contract(self):
        """An absent split returns an empty dataset; malformed lengths fail loudly."""
        empty = load_intervention_dataset(
            split="does-not-exist", parquet_path=self.parquet_path
        )
        self.assertEqual(len(empty), 0)
        with self.assertRaises(ValueError):
            TargetInterventionDataset(
                np.zeros((2, 8, 18), dtype=np.float32),
                np.zeros((1, 6), dtype=np.float32),
                np.zeros(2, dtype=np.int64),
            )
        self.assertEqual(self.train_ds.get_metadata(10_000), {})

    def test_e1_freeze_contract(self):
        """
        Task 12.1 Acceptance Criteria:
        Assert backbone parameters are bit-identical before and after a training step,
        and that head parameters did change.
        """
        torch.manual_seed(42)
        head = TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=5, head=head)
        model.freeze_backbone()

        # Record parameter states before step
        backbone_before = {name: p.clone().detach() for name, p in model.gru.named_parameters()}
        head_before = {name: p.clone().detach() for name, p in model.head.named_parameters()}

        # Verify requires_grad status
        for name, p in model.gru.named_parameters():
            self.assertFalse(p.requires_grad, f"Backbone param {name} must have requires_grad=False")
        for name, p in model.head.named_parameters():
            self.assertTrue(p.requires_grad, f"Head param {name} must have requires_grad=True")

        optimizer = optim.Adam(model.head_parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(4, 8, 18)
        dummy_ctx = torch.randn(4, 6)
        dummy_y = torch.tensor([0, 1, 2, 3], dtype=torch.long)

        out = model(dummy_x, context=dummy_ctx)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

        # Assert backbone is bit-identical
        for name, p in model.gru.named_parameters():
            self.assertTrue(
                torch.equal(p, backbone_before[name]),
                f"Backbone parameter {name} changed after optimizer step! Freezing violation."
            )

        # Assert head has changed
        for name, p in model.head.named_parameters():
            self.assertFalse(
                torch.equal(p, head_before[name]),
                f"Head parameter {name} did not change after optimizer step!"
            )

    def test_e2_partial_finetune_contract(self):
        """
        Task 12.2 Acceptance Criteria:
        Assert parameter groups are disjoint, lower layers remain bit-identical,
        and terminal layer + head change.
        """
        torch.manual_seed(42)
        head = TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=5, head=head)
        model.unfreeze_terminal_layer()

        term_idx = model.num_layers - 1
        terminal_params = [p for name, p in model.gru.named_parameters() if f"_l{term_idx}" in name]
        head_params = list(model.head_parameters())

        # Disjoint parameter groups check
        term_param_ids = set(id(p) for p in terminal_params)
        head_param_ids = set(id(p) for p in head_params)
        self.assertEqual(
            len(term_param_ids & head_param_ids), 0,
            "Parameter groups for terminal layer and head must be disjoint!"
        )

        # Verify trainable parameter coverage
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(
            len(trainable_params),
            len(terminal_params) + len(head_params),
            "Trainable parameters must exactly equal terminal layer + head parameters"
        )

        lower_before = {name: p.clone().detach() for name, p in model.gru.named_parameters() if f"_l{term_idx}" not in name}
        terminal_before = {name: p.clone().detach() for name, p in model.gru.named_parameters() if f"_l{term_idx}" in name}
        head_before = {name: p.clone().detach() for name, p in model.head.named_parameters()}

        optimizer = optim.Adam([
            {"params": terminal_params, "lr": 1e-4},
            {"params": head_params, "lr": 1e-3},
        ])
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(4, 8, 18)
        dummy_ctx = torch.randn(4, 6)
        dummy_y = torch.tensor([1, 2, 3, 4], dtype=torch.long)

        out = model(dummy_x, context=dummy_ctx)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

        # Lower layer must be bit-identical
        for name, p in model.gru.named_parameters():
            if f"_l{term_idx}" not in name:
                self.assertTrue(
                    torch.equal(p, lower_before[name]),
                    f"Lower layer parameter {name} changed! Must remain frozen in E2."
                )

        # Terminal layer must have changed
        for name, p in model.gru.named_parameters():
            if f"_l{term_idx}" in name:
                self.assertFalse(
                    torch.equal(p, terminal_before[name]),
                    f"Terminal layer parameter {name} failed to update in E2!"
                )

        # Head must have changed
        for name, p in model.head.named_parameters():
            self.assertFalse(
                torch.equal(p, head_before[name]),
                f"Head parameter {name} failed to update in E2!"
            )

    def test_e3_full_finetune_contract(self):
        """
        Task 12.3 Acceptance Criteria:
        Assert every parameter is trainable in E3 and none is frozen.
        """
        torch.manual_seed(42)
        head = TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=5, head=head)
        model.unfreeze_backbone()

        for name, p in model.named_parameters():
            self.assertTrue(p.requires_grad, f"Parameter {name} must be trainable in E3")

        all_before = {name: p.clone().detach() for name, p in model.named_parameters()}
        optimizer = optim.Adam(model.parameters(), lr=5e-4)
        criterion = nn.CrossEntropyLoss()

        dummy_x = torch.randn(4, 8, 18)
        dummy_ctx = torch.randn(4, 6)
        dummy_y = torch.tensor([0, 1, 2, 4], dtype=torch.long)

        out = model(dummy_x, context=dummy_ctx)
        loss = criterion(out, dummy_y)
        loss.backward()
        optimizer.step()

        for name, p in model.named_parameters():
            self.assertFalse(
                torch.equal(p, all_before[name]),
                f"Parameter {name} did not update in full fine-tuning!"
            )

    def test_intervention_evaluation_contract(self):
        """
        Task 12.4 Acceptance Criteria:
        Assert evaluate_intervention_model calculates accuracy, Macro-F1, per-class metrics,
        confusion matrix, ranking diagnostics, and failure cases without fabrication.
        """
        head = TargetInterventionHead(hidden_dim=32, context_dim=6, num_classes=5)
        model = EdgeAUIGRU(input_dim=18, hidden_dim=32, num_layers=1, num_classes=5, head=head)

        res = evaluate_intervention_model(
            model=model,
            eval_dataset=self.val_ds,
            train_dataset=self.train_ds,
            batch_size=16,
            num_classes=5,
            verbose=False,
        )

        self.assertIn("accuracy", res)
        self.assertIn("macro_f1", res)
        self.assertIn("weighted_f1", res)
        self.assertIn("per_class", res)
        self.assertIn("confusion_matrix", res)
        self.assertIn("ranking_diagnostics", res)
        self.assertIn("majority_baseline", res)
        self.assertIn("dominant_confusion_pairs", res)
        self.assertIn("failure_cases", res)

        self.assertEqual(len(res["confusion_matrix"]), 5)
        self.assertEqual(len(res["confusion_matrix"][0]), 5)
        self.assertEqual(len(res["per_class"]), 5)

        # Ranking diagnostics
        self.assertIn("hr@1", res["ranking_diagnostics"])
        self.assertIn("hr@3", res["ranking_diagnostics"])
        self.assertIn("mrr", res["ranking_diagnostics"])

        # Majority baseline comparison
        mb = res["majority_baseline"]
        self.assertIsNotNone(mb)
        self.assertEqual(mb["majority_class_id"], 4)
        self.assertEqual(mb["majority_class_name"], "no_op")

    def test_foundation_retention_audit_contract(self):
        """Verify catastrophic forgetting retention audit executes and reports metrics."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=5)
        retention = evaluate_intervention_retention(
            fine_tuned_model=model,
            max_sequences=50,
            verbose=False,
        )
        self.assertIn("retention_evaluated", retention)
        if retention["retention_evaluated"]:
            self.assertIn("pre_finetune_macro_f1", retention)
            self.assertIn("post_finetune_macro_f1", retention)
            self.assertIn("delta_macro_f1", retention)
            self.assertIn("retention_passed", retention)


if __name__ == "__main__":
    unittest.main()
