"""
test_intervention_head_export.py
Comprehensive export test suite for the TargetInterventionHead.
Validates the intervention-head ONNX export contract (Task E.4, ADR-006).
"""

import os
import sys
import pytest
import torch
import numpy as np
import onnx
import onnxruntime as ort

# Add src to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

try:
    from export import export_intervention_head, validate_intervention_graph, validate_intervention_parity
except ImportError:
    pass  # Will fail in tests if not found

try:
    from training import (
        EdgeAUIGRU,
        TargetInterventionHead,
        MICROTENSOR_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_CLASSES
    )
except ImportError:
    pass

try:
    from data_manager import find_project_root
except ImportError:
    def find_project_root():
        return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_checkpoint_path():
    root = find_project_root()
    return os.path.join(root, "models", "intervention_head_e3.pth")


@pytest.fixture
def temp_output_dir(tmp_path):
    return str(tmp_path)


def test_export_produces_valid_graph(temp_output_dir):
    """Call export_intervention_head() with a temp output dir and assert both files exist and are non-empty."""
    ckpt_path = get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    res = export_intervention_head(
        model_path=ckpt_path,
        output_dir=temp_output_dir
    )
    fp32_path = res["onnx_path"]
    int8_path = res["quantized_onnx_path"]

    assert os.path.exists(fp32_path), "FP32 file should exist"
    assert os.path.getsize(fp32_path) > 0, "FP32 file should be non-empty"

    assert os.path.exists(int8_path), "INT8 file should exist"
    assert os.path.getsize(int8_path) > 0, "INT8 file should be non-empty"


def test_graph_introspection_matches_adr006(temp_output_dir):
    """Load exported FP32 ONNX graph, assert input/output names and shapes match ADR-006."""
    ckpt_path = get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    res = export_intervention_head(
        model_path=ckpt_path,
        output_dir=temp_output_dir
    )
    fp32_path = res["onnx_path"]

    graph = onnx.load(fp32_path)

    input_names = [inp.name for inp in graph.graph.input]
    assert input_names == ["sequence_input", "context_input"], f"Expected specific input names, got {input_names}"

    output_names = [out.name for out in graph.graph.output]
    assert output_names == ["intervention_logits"], f"Expected specific output names, got {output_names}"

    # Check output dimension last axis is 5
    out_shape = []
    for d in graph.graph.output[0].type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out_shape.append(d.dim_value)
        else:
            out_shape.append(d.dim_param)

    assert out_shape[-1] == 5, f"Expected output dimension last axis to be 5, got {out_shape}"


def test_pytorch_onnx_numerical_parity(temp_output_dir):
    """Create model, load E3 checkpoint, run fixed-seed input through PyTorch and ONNX, assert np.allclose."""
    ckpt_path = get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    # Export to get the ONNX path
    res = export_intervention_head(
        model_path=ckpt_path,
        output_dir=temp_output_dir
    )
    fp32_path = res["onnx_path"]

    # Initialize PyTorch model with TargetInterventionHead
    head = TargetInterventionHead(hidden_dim=HIDDEN_DIM, context_dim=6, num_classes=5)
    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=NUM_CLASSES,
        head=head
    )

    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    torch.manual_seed(42)
    dummy_sequence = torch.randn(1, 10, MICROTENSOR_DIM)
    dummy_context = torch.randn(1, 6)

    with torch.no_grad():
        pt_out = model(dummy_sequence, context=dummy_context).numpy()

    session = ort.InferenceSession(fp32_path, providers=["CPUExecutionProvider"])
    inputs = {
        "sequence_input": dummy_sequence.numpy(),
        "context_input": dummy_context.numpy()
    }
    onnx_out = session.run(None, inputs)[0]

    np.testing.assert_allclose(pt_out, onnx_out, atol=1e-5)


def test_quantized_graph_loads_and_runs(temp_output_dir):
    """Load INT8 graph with onnxruntime, run inference, assert output shape is (1, 5)."""
    ckpt_path = get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    res = export_intervention_head(
        model_path=ckpt_path,
        output_dir=temp_output_dir
    )
    int8_path = res["quantized_onnx_path"]

    session = ort.InferenceSession(int8_path, providers=["CPUExecutionProvider"])

    np.random.seed(42)
    dummy_sequence = np.random.randn(1, 10, MICROTENSOR_DIM).astype(np.float32)
    dummy_context = np.random.randn(1, 6).astype(np.float32)

    inputs = {
        "sequence_input": dummy_sequence,
        "context_input": dummy_context
    }

    onnx_out = session.run(None, inputs)[0]
    assert onnx_out.shape == (1, 5), f"Expected output shape (1, 5), got {onnx_out.shape}"


def test_foundation_artifacts_untouched(temp_output_dir):
    """Record foundation file sizes before export, export intervention, assert foundation sizes unchanged."""
    root = find_project_root()
    model_onnx_path = os.path.join(root, "models", "model.onnx")
    foundational_gru_path = os.path.join(root, "models", "foundational_gru.pth")

    sizes_before = {}
    if os.path.exists(model_onnx_path):
        sizes_before["model.onnx"] = os.path.getsize(model_onnx_path)
    if os.path.exists(foundational_gru_path):
        sizes_before["foundational_gru.pth"] = os.path.getsize(foundational_gru_path)

    ckpt_path = get_checkpoint_path()
    if os.path.exists(ckpt_path):
        export_intervention_head(
            model_path=ckpt_path,
            output_dir=temp_output_dir
        )

    if "model.onnx" in sizes_before:
        assert os.path.exists(model_onnx_path)
        assert os.path.getsize(model_onnx_path) == sizes_before["model.onnx"]

    if "foundational_gru.pth" in sizes_before:
        assert os.path.exists(foundational_gru_path)
        assert os.path.getsize(foundational_gru_path) == sizes_before["foundational_gru.pth"]


def test_artifact_sizes_within_budget(temp_output_dir):
    """Assert individual intervention FP32 < 200KB and report combined payload."""
    ckpt_path = get_checkpoint_path()
    if not os.path.exists(ckpt_path):
        pytest.skip(f"Checkpoint not found at {ckpt_path}")

    res = export_intervention_head(
        model_path=ckpt_path,
        output_dir=temp_output_dir
    )
    fp32_path = res["onnx_path"]

    size_fp32 = os.path.getsize(fp32_path)
    assert size_fp32 < 200 * 1024, f"Intervention FP32 artifact too large: {size_fp32} bytes"

    root = find_project_root()
    model_onnx_path = os.path.join(root, "models", "model.onnx")
    size_foundation = os.path.getsize(model_onnx_path) if os.path.exists(model_onnx_path) else 0

    print(f"Measured intervention FP32 size: {size_fp32} bytes")
    if size_foundation > 0:
        print(f"Measured foundation FP32 size: {size_foundation} bytes")
        print(f"Measured combined payload size: {size_fp32 + size_foundation} bytes")
