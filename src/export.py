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
    from training import EdgeAUIGRU, FEATURE_NAMES, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES
except ImportError:
    from src.training import EdgeAUIGRU, FEATURE_NAMES, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES

try:
    from data_manager import find_project_root
except ImportError:
    try:
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
        input_dim=len(FEATURE_NAMES), 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    )
    model.load_state_dict(torch.load(resolved_model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Dummy input for tracing (batch_size=1, seq_len=8, features)
    dummy_input = torch.randn(1, 8, len(FEATURE_NAMES))

    # Export to ONNX using stable TorchScript backend for dynamic quantization compatibility
    onnx_path = os.path.join(resolved_output_dir, "model.onnx")
    print(f"[Export] Exporting graph to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
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
    benchmark_metrics["onnx_path"] = onnx_path
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Quantization & ONNX Export")
    parser.add_argument("--model-path", type=str, default=None, help="Path to foundational_gru.pth")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save ONNX artifacts")
    args = parser.parse_args()

    export_and_quantize(model_path=args.model_path, output_dir=args.output_dir)
