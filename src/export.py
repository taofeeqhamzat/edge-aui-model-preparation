"""
export.py
Edge Quantization & Export for the Edge-AUI Framework.
Compiles the PyTorch model to .onnx and applies INT8 Post-Training Quantization.
"""

import os
import torch
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from training import EdgeAUIGRU, FEATURE_NAMES, HIDDEN_DIM, NUM_LAYERS, NUM_CLASSES
import time
import onnxruntime as ort
import numpy as np

def export_and_quantize(model_path="../models/foundational_gru.pth", output_dir="../models"):
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run training first.")
        return

    print("Loading PyTorch model...")
    model = EdgeAUIGRU(
        input_dim=len(FEATURE_NAMES), 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    )
    # Use weights_only=True for safety
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()

    # Dummy input for tracing (batch_size=1, seq_len=8, features)
    dummy_input = torch.randn(1, 8, len(FEATURE_NAMES))

    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    print(f"Exporting to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "seq_len"}, "output": {0: "batch_size"}}
    )

    # Quantize to INT8
    quantized_onnx_path = os.path.join(output_dir, "model_int8.onnx")
    print(f"Applying dynamic INT8 quantization to {quantized_onnx_path}...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_onnx_path,
        weight_type=QuantType.QUInt8
    )

    print("Export and Quantization Complete.")
    
    # Assertions and benchmarking
    benchmark_model(quantized_onnx_path, dummy_input.numpy())

def benchmark_model(onnx_path, dummy_input_np):
    print("\n--- Verifying Constraints ---")
    file_size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"Quantized Model Size: {file_size_mb:.3f} MB")
    
    assert file_size_mb < 20.0, f"Model memory footprint ({file_size_mb:.3f} MB) exceeds 20MB limit!"
    
    # Benchmark Latency
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
        
    avg_latency = np.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    
    print(f"Average Latency: {avg_latency:.3f} ms")
    print(f"P95 Latency: {p95_latency:.3f} ms")
    
    assert avg_latency < 50.0, f"Average latency ({avg_latency:.3f} ms) exceeds 50ms limit!"
    print("All architectural constraints met successfully.")

if __name__ == "__main__":
    export_and_quantize()
