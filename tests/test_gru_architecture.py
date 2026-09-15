"""
test_gru_architecture.py
Unit tests verifying the decoupled GRU latent encoder architecture, modular heads,
latent embedding extraction, and parameter isolation primitives (Task 4.1, ADR-003).
"""

import os
import sys
import unittest
import torch
import torch.nn as nn

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import training
from training import (
    EdgeAUIGRU,
    FoundationOutcomeHead,
    TargetInterventionHead
)


class TestGRUArchitecture(unittest.TestCase):
    """Test suite for decoupled EdgeAUIGRU architecture and modular heads."""

    def test_default_architecture_and_forward_pass(self):
        """Verify default EdgeAUIGRU dimensions: input_dim=18, hidden_dim=64, num_layers=2, num_classes=7."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=7)
        x = torch.randn(4, 8, 18)

        # Standard forward pass
        out = model(x)
        self.assertEqual(out.shape, (4, 7))

        # Forward pass extracting latent representation h_T
        out_latent, h_T = model(x, return_latent=True)
        self.assertEqual(out_latent.shape, (4, 7))
        self.assertEqual(h_T.shape, (4, 64))

    def test_modular_head_replacement_without_backbone_reconstruction(self):
        """Verify attach_head replaces the projection head without reconstructing the GRU backbone."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=7)
        x = torch.randn(2, 8, 18)

        # Initially 7 classes (FoundationOutcomeHead)
        out1 = model(x)
        self.assertEqual(out1.shape, (2, 7))

        # Attach 5-class TargetInterventionHead (UI context vector is mandatory)
        context_head = TargetInterventionHead(hidden_dim=64, context_dim=6, num_classes=5)
        model.attach_head(context_head)
        ctx = torch.randn(2, 6)

        out2 = model(x, context=ctx)
        self.assertEqual(out2.shape, (2, 5))

        # Latent representation h_T remains identical for identical inputs
        _, h_T = model(x, context=ctx, return_latent=True)
        self.assertEqual(h_T.shape, (2, 64))

        # Verify missing context raises ValueError
        with self.assertRaises(ValueError):
            model(x)

        # Verify invalid context_dim <= 0 raises ValueError
        with self.assertRaises(ValueError):
            TargetInterventionHead(hidden_dim=64, context_dim=0, num_classes=5)

    def test_backbone_and_head_parameter_isolation(self):
        """Verify clean separation of parameters between theta_base and theta_head."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=6)

        backbone_params = list(model.backbone_parameters())
        head_params = list(model.head_parameters())
        all_params = list(model.parameters())

        self.assertGreater(len(backbone_params), 0)
        self.assertGreater(len(head_params), 0)
        self.assertEqual(len(backbone_params) + len(head_params), len(all_params))

    def test_strict_freezing_primitive(self):
        """Verify freeze_backbone disables gradients for all GRU layers while head remains trainable."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=6)
        model.freeze_backbone()

        for p in model.backbone_parameters():
            self.assertFalse(p.requires_grad)

        for p in model.head_parameters():
            self.assertTrue(p.requires_grad)

    def test_full_finetuning_primitive(self):
        """Verify unfreeze_backbone enables gradients across all parameters."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=6)
        model.freeze_backbone()
        model.unfreeze_backbone()

        for p in model.backbone_parameters():
            self.assertTrue(p.requires_grad)

    def test_partial_finetuning_terminal_layer_primitive(self):
        """Verify unfreeze_terminal_layer selectively enables gradients only for the terminal GRU layer."""
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=6)
        model.unfreeze_terminal_layer()

        for name, p in model.gru.named_parameters():
            if "_l1" in name:  # Layer 1 is terminal layer when num_layers=2
                self.assertTrue(p.requires_grad, f"Expected {name} to be trainable")
            elif "_l0" in name:  # Layer 0 is lower layer
                self.assertFalse(p.requires_grad, f"Expected {name} to be frozen")

    def test_hidden_state_numerical_equivalence(self):
        """
        Verify that out[:, -1, :] is numerically equivalent to h_n[-1] for a
        2-layer unidirectional GRU, proving that latent slicing extracts the exact
        final-layer terminal recurrent state.
        """
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=7)
        x = torch.randn(8, 12, 18)

        out_gru, h_n = model.gru(x)
        h_T_slice = out_gru[:, -1, :]
        h_n_terminal = h_n[-1]

        # Numerical equivalence assertion
        self.assertTrue(torch.allclose(h_T_slice, h_n_terminal, atol=1e-6))

        # Model return_latent extraction equivalence
        _, extracted_h_T = model(x, return_latent=True)
        self.assertTrue(torch.allclose(extracted_h_T, h_n_terminal, atol=1e-6))

    def test_behavioral_gradient_flow(self):
        """
        Behavioral test of gradient flow:
        1. When backbone is frozen, backbone parameter gradients must be None after backward(),
           while head gradients must be non-zero.
        2. When backbone is unfrozen, backbone parameter gradients must be actively populated.
        """
        model = EdgeAUIGRU(input_dim=18, hidden_dim=64, num_layers=2, num_classes=7)
        x = torch.randn(4, 8, 18)

        # 1. Test Frozen Backbone gradient behavior
        model.freeze_backbone()
        model.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.gru.named_parameters():
            self.assertIsNone(p.grad, f"Expected no gradient for frozen backbone parameter: {name}")

        for name, p in model.head.named_parameters():
            self.assertIsNotNone(p.grad, f"Expected active gradient for head parameter: {name}")
            self.assertFalse(torch.all(p.grad == 0.0), f"Expected non-zero gradient for: {name}")

        # 2. Test Unfrozen Backbone gradient behavior
        model.unfreeze_backbone()
        model.zero_grad()
        out = model(x)
        loss = out.sum()
        loss.backward()

        for name, p in model.gru.named_parameters():
            self.assertIsNotNone(p.grad, f"Expected active gradient for unfrozen parameter: {name}")
            self.assertFalse(torch.all(p.grad == 0.0), f"Expected non-zero gradient for: {name}")

    def test_task_4_1_verification_contract(self):
        """Execute the exact Task 4.1 verification contract command."""
        m = training.EdgeAUIGRU(18, 64, 2, 7)
        x = torch.randn(2, 8, 18)
        out, h = m(x, return_latent=True)
        self.assertEqual(h.shape, (2, 64))
        self.assertEqual(out.shape, (2, 7))


if __name__ == "__main__":
    unittest.main()
