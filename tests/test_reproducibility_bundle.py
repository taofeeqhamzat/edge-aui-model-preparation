import os
import sys
import json
import pytest
from pathlib import Path
import numpy as np

try:
    from training.data_manager import find_project_root
except ImportError:
    def find_project_root():
        current_dir = Path(__file__).resolve().parent
        while current_dir.name != "model-preparation" and current_dir.parent != current_dir:
            current_dir = current_dir.parent
        return current_dir

sys.path.insert(0, str(find_project_root()))

from src.reproduce import assemble_bundle, hash_file

@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory):
    temp_dir = tmp_path_factory.mktemp("bundles")
    bundle_path = temp_dir / "v1.0.0"
    
    result_path = assemble_bundle('e3', 'v1.0.0', str(bundle_path))
    return Path(result_path)

def test_bundle_structure(bundle_dir):
    assert (bundle_dir / "bundle.json").exists()
    assert (bundle_dir / "manifest.json").exists()
    
    with open(bundle_dir / "bundle.json", 'r') as f:
        data = json.load(f)
        
    for key in ["model_artifact", "model_artifact_int8", "model_checkpoint"]:
        if data.get(key):
            assert (bundle_dir / data[key]).exists()

def test_bundle_json_schema(bundle_dir):
    with open(bundle_dir / "bundle.json", 'r') as f:
        data = json.load(f)
        
    required_keys = [
        "experiment", "model_artifact", "model_checkpoint", 
        "model_hash_sha256", "onnx_hash_sha256", "dataset_version", 
        "dataset_content_hash", "code_commit", "seed", 
        "input_contract", "output_contract", "intervention_vocabulary", 
        "environment", "metrics", "consumed_by"
    ]
    
    for key in required_keys:
        assert key in data, f"Missing required key: {key}"

def test_bundle_hashes_match(bundle_dir):
    with open(bundle_dir / "bundle.json", 'r') as f:
        data = json.load(f)
        
    ckpt_name = data.get("model_checkpoint")
    if ckpt_name and (bundle_dir / ckpt_name).exists():
        assert hash_file(bundle_dir / ckpt_name) == data.get("model_hash_sha256")
        
    onnx_name = data.get("model_artifact")
    if onnx_name and (bundle_dir / onnx_name).exists():
        assert hash_file(bundle_dir / onnx_name) == data.get("onnx_hash_sha256")

def test_bundle_onnx_loads(bundle_dir):
    with open(bundle_dir / "bundle.json", 'r') as f:
        data = json.load(f)
        
    onnx_name = data.get("model_artifact")
    if not onnx_name or not (bundle_dir / onnx_name).exists():
        pytest.skip("ONNX model not found in bundle")
        
    import onnxruntime as ort
    session = ort.InferenceSession(str(bundle_dir / onnx_name))
    
    input_details = session.get_inputs()
    seq_name = input_details[0].name
    ctx_name = input_details[1].name
    
    dummy_seq = np.random.randn(1, 5, 18).astype(np.float32)
    dummy_ctx = np.random.randn(1, 6).astype(np.float32)
    
    outputs = session.run(None, {seq_name: dummy_seq, ctx_name: dummy_ctx})
    assert outputs[0].shape == (1, 5)

def test_bundle_contract_matches_adr006(bundle_dir):
    with open(bundle_dir / "bundle.json", 'r') as f:
        data = json.load(f)
        
    inputs = data["input_contract"]
    outputs = data["output_contract"]
    
    assert "sequence_input" in inputs
    assert inputs["sequence_input"]["shape"][-1] == 18
    
    assert "context_input" in inputs
    assert inputs["context_input"]["shape"][-1] == 6
    
    assert "intervention_logits" in outputs
    assert outputs["intervention_logits"]["shape"][-1] == 5
