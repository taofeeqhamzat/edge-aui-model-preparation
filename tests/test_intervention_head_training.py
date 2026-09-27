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
import unittest
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
from target_dataset import (
    TargetInterventionDataset,
    load_intervention_dataset,
    get_intervention_class_weights,
    INTERVENTION_VOCABULARY,
    INTERVENTION_TO_ID,
)


class TestInterventionHeadTraining(unittest.TestCase):
    """Test suite covering Phase D tasks (12.1, 12.2, 12.3, 12.4)."""

    @classmethod
    def setUpClass(cls):
        cls.train_ds = load_intervention_dataset(version="v1.0.0", split="train")
        cls.val_ds = load_intervention_dataset(version="v1.0.0", split="val")
        cls.test_ds = load_intervention_dataset(version="v1.0.0", split="test")
        cls.class_weights = get_intervention_class_weights(version="v1.0.0")

    def test_dataset_loader_contract(self):
        """Assert dataset partitions match expected example counts and batch tensor shapes."""
        self.assertEqual(len(self.train_ds), 208, "Expected 208 training examples in v1.0.0")
        self.assertEqual(len(self.val_ds), 51, "Expected 51 validation examples in v1.0.0")
        self.assertEqual(len(self.test_ds), 30, "Expected 30 test examples in v1.0.0")

        # Test index item
        seq, ctx, tgt = self.train_ds[0]
        self.assertEqual(seq.shape, (8, 18), "MicroTensor sequence shape must be (8, 18)")
        self.assertEqual(ctx.shape, (6,), "Context vector shape must be (6,)")
        self.assertIsInstance(tgt.item(), int)
        self.assertTrue(0 <= tgt.item() < 5, "Target must be in range 0..4")

        # Test class weights
        self.assertEqual(self.class_weights.shape, (5,), "Class weights must be 5-dimensional")
        self.assertTrue((self.class_weights > 0).all(), "All class weights must be positive")

        # Test metadata
        meta = self.train_ds.get_metadata(0)
        self.assertIn("session_id", meta)
        self.assertIn("task_id", meta)

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
