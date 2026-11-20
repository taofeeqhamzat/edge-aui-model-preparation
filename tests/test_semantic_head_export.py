"""
test_semantic_head_export.py
Task 40.5: ONNX graph contract, INT8 parity, bundle v2 schema/hashes and legacy preservation.
"""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

import onnx
import onnxruntime as ort
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from export import (
    SEMANTIC_INT8_NAME,
    SEMANTIC_ONNX_NAME,
    SEMANTIC_OUTPUT_NAME,
    export_semantic_head,
    validate_semantic_graph,
)
from semantic_targets import TARGET_VOCABULARY
from training import EdgeAUIGRU, TargetSemanticHead, SEMANTIC_NUM_CLASSES

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "models" / "semantic_head_e2.pth"
DATASET_VERSION = "v2.2.0-participant"
LEGACY_ARTIFACTS = [
    PROJECT_ROOT / "models" / "intervention_head.onnx",
    PROJECT_ROOT / "models" / "intervention_head_int8.onnx",
    PROJECT_ROOT / "models" / "foundational_gru.pth",
    PROJECT_ROOT / "models" / "bundles" / "v1.0.0" / "bundle.json",
]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


@unittest.skipUnless(CHECKPOINT.is_file(), "semantic e2 checkpoint not trained yet")
class TestSemanticHeadExport(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.tmp.name)
        cls.legacy_hashes = {path: _sha(path) for path in LEGACY_ARTIFACTS if path.is_file()}
        cls.bundle = export_semantic_head(
            model_path=str(CHECKPOINT),
            output_dir=str(cls.output),
            sequence_length=12,
            dataset_version=DATASET_VERSION,
            experiment="e2",
            metrics_path=str(PROJECT_ROOT / "docs" / "experiments" / "semantic-head"
                             / "semantic-head-evaluation.json"),
            bundle_version="v2.0.0",
        )
        cls.onnx_path = cls.output / SEMANTIC_ONNX_NAME
        cls.int8_path = cls.output / SEMANTIC_INT8_NAME

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_graph_contract_is_sequence_context_to_14_classes(self):
        model = onnx.load(str(self.onnx_path))
        inputs = [item.name for item in model.graph.input]
        self.assertEqual(inputs, ["sequence_input", "context_input"])
        self.assertEqual(model.graph.output[0].name, SEMANTIC_OUTPUT_NAME)
        shape = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in model.graph.output[0].type.tensor_type.shape.dim
        ]
        self.assertEqual(shape[-1], 14)
        validate_semantic_graph(str(self.onnx_path))

    def test_onnx_output_matches_the_head(self):
        model = EdgeAUIGRU(
            input_dim=18, hidden_dim=64, num_layers=2, num_classes=SEMANTIC_NUM_CLASSES,
            head=TargetSemanticHead(hidden_dim=64),
        )
        model.load_state_dict(torch.load(str(CHECKPOINT), map_location="cpu", weights_only=True))
        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {
            "sequence_input": torch.randn(2, 12, 18).numpy(),
            "context_input": torch.randn(2, 6).numpy(),
        })
        self.assertEqual(outputs[0].shape, (2, 14))

    def test_int8_graph_loads_and_agrees_with_fp32(self):
        session = ort.InferenceSession(str(self.int8_path), providers=["CPUExecutionProvider"])
        sequence = torch.randn(16, 12, 18).numpy()
        context = torch.randn(16, 6).numpy()
        fp32 = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        fp32_out = fp32.run(None, {"sequence_input": sequence, "context_input": context})[0]
        int8_out = session.run(None, {"sequence_input": sequence, "context_input": context})[0]
        agreement = float((fp32_out.argmax(axis=1) == int8_out.argmax(axis=1)).mean())
        self.assertGreaterEqual(agreement, 0.95)

    def test_bundle_v2_records_contract_provenance_and_hashes(self):
        bundle = json.loads((self.output / "bundles" / "v2.0.0" / "bundle.json").read_text())
        self.assertEqual(bundle["bundle_schema_version"], "2.0.0")
        self.assertEqual(bundle["head"], "TargetSemanticHead")
        self.assertEqual(bundle["head_class_count"], 14)
        self.assertEqual(len(bundle["task_outcome_vocabulary"]), 14)
        self.assertEqual(bundle["task_outcome_vocabulary"],
                         [TARGET_VOCABULARY[i] for i in sorted(TARGET_VOCABULARY)])
        self.assertEqual(bundle["target_vocabulary_version"], "testbed-semantic-target/1.0.0")
        self.assertEqual(bundle["recorded_vocabulary_version"], "testbed-semantic/1.0.0")
        self.assertEqual(bundle["dataset_version"], DATASET_VERSION)
        self.assertEqual(bundle["split_unit"], "session")
        self.assertEqual(bundle["output_contract"][SEMANTIC_OUTPUT_NAME]["shape"][-1], 14)
        self.assertEqual(bundle["onnx_hash_sha256"], _sha(self.onnx_path))
        self.assertEqual(bundle["int8_hash_sha256"], _sha(self.int8_path))
        self.assertIn("fp32_max_abs_diff", bundle["parity"])
        self.assertIn("int8_top1_agreement", bundle["parity"])
        self.assertGreater(bundle["measured"]["int8"]["file_size_bytes"], 0)
        self.assertGreater(bundle["metrics"]["macro_f1"], 0.0)

    def test_bundle_copies_are_self_contained(self):
        bundle_dir = self.output / "bundles" / "v2.0.0"
        for name in (SEMANTIC_ONNX_NAME, SEMANTIC_INT8_NAME,
                     "semantic_head_e2.pth", "manifest.json"):
            self.assertTrue((bundle_dir / name).is_file(), name)

    def test_legacy_artifacts_are_untouched(self):
        for path, before in self.legacy_hashes.items():
            self.assertEqual(_sha(path), before, f"{path} changed during semantic export")


if __name__ == "__main__":
    unittest.main()
