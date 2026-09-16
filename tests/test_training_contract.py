"""
test_training_contract.py
Unit tests verifying the training interface contract, 18-D model input dimension,
and end-to-end training smoke test with Parquet-backed sequences.
"""

import os
import sys
import unittest
import torch

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from training import EdgeAUIGRU, train_foundation_model, load_foundation_dataset
from preprocessing import MICROTENSOR_DIM


class TestTrainingContract(unittest.TestCase):
    """Verify training dimension contracts and dataset interfaces."""

    def test_model_input_dimension_is_18(self):
        """Ensure EdgeAUIGRU accepts (batch, seq_len, 18) inputs without dimension mismatch."""
        self.assertEqual(MICROTENSOR_DIM, 18)
        model = EdgeAUIGRU(input_dim=MICROTENSOR_DIM, hidden_dim=32, num_layers=2, num_classes=7)
        
        # Forward pass with 18-D input
        dummy_x = torch.randn(4, 8, 18)
        out = model(dummy_x)
        self.assertEqual(out.shape, (4, 7))

    def test_load_foundation_dataset_contract(self):
        """Verify load_foundation_dataset produces 18-D tensors and valid outcome targets."""
        dataset = load_foundation_dataset(max_sequences=50, split="train")
        self.assertGreater(len(dataset), 0)

        x, y = dataset[0]
        self.assertEqual(x.ndim, 2)
        self.assertEqual(x.shape[1], 18, f"Expected 18 features, got {x.shape[1]}")
        self.assertIsInstance(y.item(), int)

    def test_training_smoke_test(self):
        """Run a 2-epoch smoke test training run on a small sequence subset."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            res = train_foundation_model(
                epochs=2,
                batch_size=16,
                max_sequences=32,
                output_dir=tmp_dir,
                verbose=False
            )
            self.assertIsNotNone(res["model"])
            self.assertEqual(len(res["history"]["loss"]), 2)
            self.assertEqual(len(res["history"]["accuracy"]), 2)
            self.assertTrue(os.path.isfile(res["model_path"]))
            self.assertIn("val_evaluation", res)
            if res["val_evaluation"] is not None:
                self.assertIn("macro_f1", res["val_evaluation"])
                self.assertIn("accuracy", res["val_evaluation"])
                self.assertIn("ranking_diagnostics", res["val_evaluation"])


if __name__ == "__main__":
    unittest.main()
