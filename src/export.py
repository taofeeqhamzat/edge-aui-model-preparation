"""
export.py
Edge Quantization & Export for the Edge-AUI Framework.
Compiles the PyTorch model to .onnx and applies INT8 Post-Training Quantization.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

try:
    from training import (
        EdgeAUIGRU,
        TargetInterventionHead,
        TargetSemanticHead,
        FEATURE_NAMES,
        MICROTENSOR_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_CLASSES,
        SEMANTIC_NUM_CLASSES,
    )
    from semantic_targets import (
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        RECORDED_VOCABULARY_VERSION,
        RECORDED_ONLY_OUTCOMES,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from src.training import (
        EdgeAUIGRU,
        TargetInterventionHead,
        TargetSemanticHead,
        FEATURE_NAMES,
        MICROTENSOR_DIM,
        HIDDEN_DIM,
        NUM_LAYERS,
        NUM_CLASSES,
        SEMANTIC_NUM_CLASSES,
    )
    from src.semantic_targets import (
        TARGET_VOCABULARY,
        TARGET_VOCABULARY_VERSION,
        RECORDED_VOCABULARY_VERSION,
        RECORDED_ONLY_OUTCOMES,
    )

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


# ---------------------------------------------------------------------------
# Semantic task-outcome head (Task 40.5)
# ---------------------------------------------------------------------------

SEMANTIC_OUTPUT_NAME = "task_outcome_logits"
SEMANTIC_ONNX_NAME = "task_outcome_head_v2.onnx"
SEMANTIC_INT8_NAME = "task_outcome_head_v2_int8.onnx"


def validate_semantic_graph(onnx_path: str) -> Dict[str, Any]:
    """Assert the (sequence, context) -> 14-class task-outcome graph contract."""
    model = onnx.load(onnx_path)
    input_names = [item.name for item in model.graph.input]
    outputs = model.graph.output
    if input_names != ["sequence_input", "context_input"]:
        raise AssertionError(
            f"Expected inputs ['sequence_input', 'context_input'], got {input_names}"
        )
    if len(outputs) != 1 or outputs[0].name != SEMANTIC_OUTPUT_NAME:
        raise AssertionError(
            f"Expected one output '{SEMANTIC_OUTPUT_NAME}', got "
            f"{[item.name for item in outputs]}"
        )
    shape = [
        dim.dim_value if dim.HasField("dim_value") else dim.dim_param
        for dim in outputs[0].type.tensor_type.shape.dim
    ]
    if shape[-1] != SEMANTIC_NUM_CLASSES:
        raise AssertionError(
            f"Expected {SEMANTIC_NUM_CLASSES} output classes, got {shape[-1]}"
        )
    print("[Validation] Semantic target-outcome graph structure valid.")
    return {"inputs": input_names, "outputs": [SEMANTIC_OUTPUT_NAME], "output_shape": shape}


def _semantic_session(onnx_path: str) -> ort.InferenceSession:
    return ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])


def validate_semantic_parity(
    model, onnx_path: str, sequence_length: int, atol: float = 1e-4
) -> Dict[str, float]:
    torch.manual_seed(42)
    sequence = torch.randn(4, sequence_length, MICROTENSOR_DIM)
    context = torch.randn(4, 6)
    with torch.no_grad():
        torch_out = model(sequence, context=context).numpy()
    session = _semantic_session(onnx_path)
    onnx_out = session.run(None, {
        "sequence_input": sequence.numpy(),
        "context_input": context.numpy(),
    })[0]
    max_diff = float(np.max(np.abs(torch_out - onnx_out)))
    if max_diff > atol:
        raise AssertionError(
            f"PyTorch/ONNX parity failed: max difference {max_diff:.6e} > {atol:.6e}"
        )
    print(f"[Validation] FP32 parity passed (max |diff| = {max_diff:.3e}).")
    return {"fp32_max_abs_diff": max_diff}


def validate_semantic_int8_parity(
    model, int8_path: str, sequence_length: int, min_agreement: float = 0.95
) -> Dict[str, float]:
    """Quantised parity: report logit drift and top-1 agreement, never hide them."""
    torch.manual_seed(7)
    sequence = torch.randn(32, sequence_length, MICROTENSOR_DIM)
    context = torch.randn(32, 6)
    with torch.no_grad():
        torch_logits = model(sequence, context=context).numpy()
    session = _semantic_session(int8_path)
    int8_logits = session.run(None, {
        "sequence_input": sequence.numpy(),
        "context_input": context.numpy(),
    })[0]
    max_diff = float(np.max(np.abs(torch_logits - int8_logits)))
    agreement = float(np.mean(torch_logits.argmax(axis=1) == int8_logits.argmax(axis=1)))
    if agreement < min_agreement:
        raise AssertionError(
            f"INT8 top-1 agreement {agreement:.3f} < {min_agreement:.3f}"
        )
    print(
        f"[Validation] INT8 parity: top-1 agreement {agreement:.3f}, "
        f"max |logit diff| {max_diff:.3f}."
    )
    return {"int8_top1_agreement": agreement, "int8_max_abs_diff": max_diff}


def benchmark_semantic_model(
    onnx_path: str, dummy_sequence_np: np.ndarray, dummy_context_np: np.ndarray
) -> Dict[str, float]:
    """Measure size and latency. Report actual values; make no budget claim."""
    file_size_bytes = os.path.getsize(onnx_path)
    session = _semantic_session(onnx_path)
    inputs = {
        "sequence_input": dummy_sequence_np,
        "context_input": dummy_context_np,
    }
    for _ in range(10):
        session.run(None, inputs)
    latencies = []
    for _ in range(100):
        start = time.perf_counter()
        session.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000.0)
    return {
        "file_size_bytes": file_size_bytes,
        "file_size_mb": file_size_bytes / (1024 * 1024),
        "avg_latency_ms": float(np.mean(latencies)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
    }


def _sha256_file(path: Path) -> str:
    import hashlib
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_metrics(
    metrics_path: Optional[str], experiment: str
) -> Dict[str, Any]:
    if not metrics_path:
        return {}
    path = Path(metrics_path)
    if not path.is_file():
        raise FileNotFoundError(f"Metrics report not found: {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    for arm in report.get("ablation", []):
        if arm.get("experiment") == experiment:
            return arm.get("test", {})
    print(f"[Export] Warning: no metrics arm for experiment '{experiment}' in {path}")
    return {}


def export_semantic_head(
    model_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    sequence_length: int = 12,
    opset_version: int = 17,
    dataset_version: str = "v2.2.0-participant",
    experiment: str = "e1",
    metrics_path: Optional[str] = None,
    bundle_version: str = "v2.0.0",
    seed: int = 42,
    training_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Export the 14-class semantic head, quantise it and assemble bundle v2."""
    resolved_model_path, resolved_output_dir = resolve_model_paths(
        model_path or f"models/semantic_head_{experiment}.pth", output_dir
    )
    if not os.path.exists(resolved_model_path):
        raise FileNotFoundError(
            f"[Export] Semantic checkpoint not found at '{resolved_model_path}'. "
            f"Run `python3 -m src.semantic_training --experiment {experiment}` first."
        )

    head = TargetSemanticHead(hidden_dim=HIDDEN_DIM, context_dim=6, num_classes=SEMANTIC_NUM_CLASSES)
    model = EdgeAUIGRU(
        input_dim=MICROTENSOR_DIM,
        hidden_dim=HIDDEN_DIM,
        num_layers=NUM_LAYERS,
        num_classes=SEMANTIC_NUM_CLASSES,
        head=head,
    )
    model.load_state_dict(torch.load(resolved_model_path, map_location="cpu", weights_only=True))
    model.eval()

    dummy_sequence = torch.randn(1, sequence_length, MICROTENSOR_DIM)
    dummy_context = torch.randn(1, 6)

    onnx_path = os.path.join(resolved_output_dir, SEMANTIC_ONNX_NAME)
    print(f"[Export] Exporting semantic graph to ONNX: {onnx_path}")
    torch.onnx.export(
        model,
        (dummy_sequence, dummy_context),
        onnx_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["sequence_input", "context_input"],
        output_names=[SEMANTIC_OUTPUT_NAME],
        dynamic_axes={
            "sequence_input": {0: "batch_size", 1: "seq_len"},
            "context_input": {0: "batch_size"},
            SEMANTIC_OUTPUT_NAME: {0: "batch_size"},
        },
        dynamo=False,
    )
    validate_semantic_graph(onnx_path)
    fp32_parity = validate_semantic_parity(model, onnx_path, sequence_length)

    int8_path = os.path.join(resolved_output_dir, SEMANTIC_INT8_NAME)
    print(f"[Export] Applying dynamic INT8 quantization: {int8_path}...")
    quantize_dynamic(model_input=onnx_path, model_output=int8_path, weight_type=QuantType.QUInt8)
    int8_parity = validate_semantic_int8_parity(model, int8_path, sequence_length)

    fp32_bench = benchmark_semantic_model(onnx_path, dummy_sequence.numpy(), dummy_context.numpy())
    int8_bench = benchmark_semantic_model(int8_path, dummy_sequence.numpy(), dummy_context.numpy())
    print(
        f"[Export] <Measured> FP32 {fp32_bench['file_size_bytes']} B / "
        f"{fp32_bench['avg_latency_ms']:.2f} ms; INT8 {int8_bench['file_size_bytes']} B / "
        f"{int8_bench['avg_latency_ms']:.2f} ms </Measured>"
    )

    root = Path(find_project_root())
    manifest_path = root / ".data" / "processed" / dataset_version / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    training_config_file = (
        Path(training_config_path)
        if training_config_path
        else Path(resolved_model_path).with_name(f"semantic_head_{experiment}_training.json")
    )
    training_config = (
        json.loads(training_config_file.read_text(encoding="utf-8"))
        if training_config_file.is_file()
        else {}
    )

    code_commit = "unknown"
    try:
        import subprocess
        code_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(root), capture_output=True, text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        pass

    environment = {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "onnx_opset": opset_version,
    }

    checkpoint = Path(resolved_model_path)
    bundle: Dict[str, Any] = {
        "bundle_schema_version": "2.0.0",
        "head": "TargetSemanticHead",
        "experiment": experiment,
        "model_artifact": SEMANTIC_ONNX_NAME,
        "model_artifact_int8": SEMANTIC_INT8_NAME,
        "model_checkpoint": checkpoint.name,
        "onnx_hash_sha256": _sha256_file(Path(onnx_path)),
        "int8_hash_sha256": _sha256_file(Path(int8_path)),
        "checkpoint_hash_sha256": _sha256_file(checkpoint),
        "dataset_version": dataset_version,
        "dataset_content_hash": manifest.get("content_hash"),
        "dataset_manifest": "manifest.json",
        "split_unit": manifest.get("split_summary", {}).get("split_unit", "session"),
        "target_vocabulary_version": TARGET_VOCABULARY_VERSION,
        "recorded_vocabulary_version": RECORDED_VOCABULARY_VERSION,
        "recorded_only_outcomes": sorted(RECORDED_ONLY_OUTCOMES),
        "task_outcome_vocabulary": [TARGET_VOCABULARY[i] for i in sorted(TARGET_VOCABULARY)],
        "head_class_count": SEMANTIC_NUM_CLASSES,
        "input_contract": {
            "sequence_input": {"shape": ["batch", "seq_len", MICROTENSOR_DIM], "dtype": "float32"},
            "context_input": {"shape": ["batch", 6], "dtype": "float32"},
        },
        "output_contract": {
            SEMANTIC_OUTPUT_NAME: {"shape": ["batch", SEMANTIC_NUM_CLASSES], "dtype": "float32"}
        },
        "sequence_length": sequence_length,
        "seed": seed,
        "code_commit": code_commit,
        "environment": environment,
        "metrics": _semantic_metrics(metrics_path, experiment),
        "parity": {**fp32_parity, **int8_parity},
        "measured": {"fp32": fp32_bench, "int8": int8_bench},
        "training_config": training_config_file.name if training_config_file.is_file() else None,
        "training": {
            key: training_config.get(key)
            for key in (
                "label_key", "lookahead_horizon_ms", "class_weights",
                "class_weights_source", "best_epoch", "learning_rate_scheme",
            )
        },
        "claim_boundary": (
            "Participant-derived engineering evidence. Session-bounded split unit; no "
            "participant identifier is approved (ADR-023), so no participant-held-out "
            "generalisation is claimed. Model accuracy is not evidence of user benefit."
        ),
        "consumed_by": ["edge-aui-framework"],
    }

    bundle_dir = Path(resolved_output_dir) / "bundles" / bundle_version
    bundle_dir.mkdir(parents=True, exist_ok=True)
    import shutil
    for source in (Path(onnx_path), Path(int8_path), checkpoint):
        shutil.copy2(source, bundle_dir / source.name)
    shutil.copy2(manifest_path, bundle_dir / "manifest.json")
    if training_config_file.is_file():
        shutil.copy2(training_config_file, bundle_dir / training_config_file.name)
    (bundle_dir / "bundle.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")

    print(f"[Export] Bundle {bundle_version} written to {bundle_dir}")
    print("[Export] Semantic head export and INT8 quantization complete.")
    return bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edge Quantization & ONNX Export")
    parser.add_argument(
        "--head", type=str, choices=["foundation", "intervention", "semantic"],
        default="foundation", help="Which head to export",
    )
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint (.pth)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save ONNX artifacts")
    parser.add_argument("--dataset-version", type=str, default="v2.2.0-participant")
    parser.add_argument("--experiment", type=str, default="e1", choices=["e1", "e2", "e3"])
    parser.add_argument("--sequence-length", type=int, default=12)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--bundle-version", type=str, default="v2.0.0")
    args = parser.parse_args()

    if args.head == "intervention":
        export_intervention_head(model_path=args.model_path, output_dir=args.output_dir)
    elif args.head == "semantic":
        export_semantic_head(
            model_path=args.model_path,
            output_dir=args.output_dir,
            sequence_length=args.sequence_length,
            dataset_version=args.dataset_version,
            experiment=args.experiment,
            metrics_path=args.metrics,
            bundle_version=args.bundle_version,
        )
    else:
        export_and_quantize(model_path=args.model_path, output_dir=args.output_dir)
