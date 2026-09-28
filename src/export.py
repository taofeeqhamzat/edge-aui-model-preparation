"""
export.py
Edge Quantization & Export for the Edge-AUI Framework.
Compiles the PyTorch model to .onnx and applies INT8 Post-Training Quantization.
"""

import os
import sys
import time
import argparse
from typing import Dict, Any, Optional
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    from training import EdgeAUIGRU, TargetInterventionHead, FEATURE_NAMES, MICROTENSOR_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES
except ImportError:
    # pyrefly: ignore [missing-import]
    from src.training import EdgeAUIGRU, TargetInterventionHead, FEATURE_NAMES, MICROTENSOR_DIM, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES

try:
    from data_manager import find_project_root
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.data_manager import find_project_root
    except ImportError:
        find_project_root = lambda: os.getcwd()


def resolve_model_paths(model_path: Optional[str] = None, output_dir: Optional[str] = None):
    proj_root = str(find_project_root())
    
    if model_path is None:
        candidates = [
            os.path.join(proj_root, "models", "foundational_gru.pth"),
            "models/foundational_gru.pth",
            "../models/foundational_gru.pth"
        ]
        resolved_model = next((c for c in candidates if os.path.exists(c)), candidates[0])
    else:
        resolved_model = os.path.abspath(model_path)

    if output_dir is None:
        resolved_output = os.path.join(proj_root, "models")
    else:
        resolved_output = os.path.abspath(output_dir)

    os.makedirs(resolved_output, exist_ok=True)
    return resolved_model, resolved_output


def export_and_quantize(
    model_path: Optional[str] = None, 
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    resolved_model_path, resolved_output_dir = resolve_model_paths(model_path, output_dir)

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"[Export] Model not found at '{resolved_model_path}'. Please run training first.")

    print(f"[Export] Loading PyTorch model from: {resolved_model_path}")
    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(resolved_model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Dummy input for tracing (batch_size=1, seq_len=8, features)
    dummy_input = torch.randn(1, 8, MICROTENSOR_DIM)

    # Export to ONNX using stable TorchScript backend for dynamic quantization compatibility
    onnx_path = os.path.join(resolved_output_dir, "model.onnx")
    print(f"[Export] Exporting graph to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        # pyrefly: ignore [bad-argument-type]
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}},
        dynamo=False
    )

    # Quantize to INT8
    quantized_onnx_path = os.path.join(resolved_output_dir, "model_int8.onnx")
    print(f"[Export] Applying dynamic INT8 quantization: {quantized_onnx_path}...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )

    print("[Export] Export and INT8 Quantization Complete.")
    
    # Assertions and benchmarking
    benchmark_metrics = benchmark_model(quantized_onnx_path, dummy_input.numpy())
    # pyrefly: ignore [unsupported-operation]
    benchmark_metrics["onnx_path"] = onnx_path
    # pyrefly: ignore [unsupported-operation]
    benchmark_metrics["quantized_onnx_path"] = quantized_onnx_path
    return benchmark_metrics


def benchmark_model(onnx_path: str, dummy_input_np: np.ndarray) -> Dict[str, float]:
    print("\n--- Verifying Edge Architectural Constraints ---")
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Quantized Model Size: {file_size_mb:.3f} MB (Limit: <20MB)")
    
    assert file_size_mb < 20.0, f"Model memory footprint ({file_size_mb:.3f} MB) exceeds 20MB limit!"
    
    # Benchmark Latency using CPU provider (representing edge execution)
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(10):
        session.run(None, {input_name: dummy_input_np})
        
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, {input_name: dummy_input_np})
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    
    print(f"Average Inference Latency: {avg_latency:.3f} ms (Limit: <50ms)")
    print(f"P95 Inference Latency:     {p95_latency:.3f} ms")
    
    assert avg_latency < 50.0, f"Average latency ({avg_latency:.3f} ms) exceeds 50ms limit!"
    print("[Verification] All edge architectural constraints met successfully.")
    
    return {
        "file_size_mb": file_size_mb,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency
    }


def validate_intervention_graph(onnx_path: str) -> Dict[str, Any]:
    model = onnx.load(onnx_path)
    inputs = model.graph.input
    outputs = model.graph.output
    
    input_names = [i.name for i in inputs]
    assert len(inputs) == 2, f"Expected 2 inputs, got {len(inputs)}"
    assert "sequence_input" in input_names, "Missing sequence_input"
    assert "context_input" in input_names, "Missing context_input"
    
    assert len(outputs) == 1, f"Expected 1 output, got {len(outputs)}"
    assert outputs[0].name == "intervention_logits", f"Expected intervention_logits, got {outputs[0].name}"
    
    # Check output dimension
    out_shape = [dim.dim_value if dim.HasField('dim_value') else dim.dim_param for dim in outputs[0].type.tensor_type.shape.dim]
    assert out_shape[-1] == 5, f"Expected last output dim to be 5, got {out_shape[-1]}"
    
    print("[Validation] Intervention graph structure valid.")
    return {"inputs": input_names, "outputs": [outputs[0].name], "output_shape": out_shape}


def validate_intervention_parity(model, onnx_path: str, sequence_length: int, atol: float = 1e-5) -> float:
    torch.manual_seed(42)
    dummy_sequence = torch.randn(1, sequence_length, MICROTENSOR_DIM)
    dummy_context = torch.randn(1, 6)
    
    # PyTorch inference
    with torch.no_grad():
        torch_out = model(dummy_sequence, context=dummy_context).numpy()
        
    # ONNX inference
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {
        "sequence_input": dummy_sequence.numpy(),
        "context_input": dummy_context.numpy()
    })[0]
    
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    assert np.allclose(torch_out, onnx_out, atol=atol), f"Parity check failed! Max diff: {max_diff}"
    print(f"[Validation] Parity check passed. Max difference: {max_diff:.6e}")
    return max_diff


def benchmark_intervention_model(onnx_path: str, dummy_sequence_np: np.ndarray, dummy_context_np: np.ndarray) -> Dict[str, float]:
    print("\n--- Verifying Edge Architectural Constraints (Intervention Head) ---")
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Quantized Model Size: {file_size_mb:.3f} MB (Limit: <20MB)")
    
    assert file_size_mb < 20.0, f"Model memory footprint ({file_size_mb:.3f} MB) exceeds 20MB limit!"
    
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    
    inputs = {
        "sequence_input": dummy_sequence_np,
        "context_input": dummy_context_np
    }
    
    # Warmup
    for _ in range(10):
        session.run(None, inputs)
        
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)  # ms
        
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))
    
    print(f"Average Inference Latency: {avg_latency:.3f} ms (Limit: <50ms)")
    print(f"P95 Inference Latency:     {p95_latency:.3f} ms")
    
    assert avg_latency < 50.0, f"Average latency ({avg_latency:.3f} ms) exceeds 50ms limit!"
    print("[Verification] All edge architectural constraints met successfully.")
    
    return {
        "file_size_mb": file_size_mb,
        "avg_latency_ms": avg_latency,
        "p95_latency_ms": p95_latency
    }


def export_intervention_head(
    model_path: Optional[str] = None, 
    output_dir: Optional[str] = None,
    sequence_length: int = 10,
    opset_version: int = 17
) -> Dict[str, Any]:
    resolved_model_path, resolved_output_dir = resolve_model_paths(model_path, output_dir)

    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(f"[Export] Model not found at '{resolved_model_path}'.")

    print(f"[Export] Loading PyTorch model from: {resolved_model_path}")
    head = TargetInterventionHead(hidden_dim=HIDDEN_DIM, context_dim=6, num_classes=5)
    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM, 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES,
        head=head
    )
    model.load_state_dict(torch.load(resolved_model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_sequence = torch.randn(1, sequence_length, MICROTENSOR_DIM)
    dummy_context = torch.randn(1, 6)

    onnx_path = os.path.join(resolved_output_dir, "intervention_head.onnx")
    print(f"[Export] Exporting graph to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_sequence, dummy_context),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["sequence_input", "context_input"],
        output_names=["intervention_logits"],
        dynamic_axes={
            "sequence_input": {0: "batch_size", 1: "seq_len"},
            "context_input": {0: "batch_size"},
            "intervention_logits": {0: "batch_size"}
        },
        dynamo=False
    )

    validate_intervention_graph(onnx_path)
    validate_intervention_parity(model, onnx_path, sequence_length)

    quantized_onnx_path = os.path.join(resolved_output_dir, "intervention_head_int8.onnx")
    print(f"[Export] Applying dynamic INT8 quantization: {quantized_onnx_path}...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )

    fp32_size = os.path.getsize(onnx_path)
    int8_size = os.path.getsize(quantized_onnx_path)
    
    print(f"[Export] Evidence Tag: <Measured> FP32 Size: {fp32_size} bytes, INT8 Size: {int8_size} bytes </Measured>")
    if int8_size >= fp32_size:
        print("[Export] INT8 quantisation did not reduce the graph size. Plan 0 observed the same pattern for the foundation model.")

    benchmark_metrics = benchmark_intervention_model(
        quantized_onnx_path, 
        dummy_sequence.numpy(), 
        dummy_context.numpy()
    )
    benchmark_metrics["onnx_path"] = onnx_path
    benchmark_metrics["quantized_onnx_path"] = quantized_onnx_path
    
    print("[Export] Intervention Head Export and INT8 Quantization Complete.")
    return benchmark_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Quantization & ONNX Export")
    parser.add_argument("--head", type=str, choices=["foundation", "intervention"], default="foundation", help="Which head to export")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save ONNX artifacts")
    args = parser.parse_args()

    if args.head == "intervention":
        export_intervention_head(model_path=args.model_path, output_dir=args.output_dir)
    else:
        export_and_quantize(model_path=args.model_path, output_dir=args.output_dir)
