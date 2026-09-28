import os
import sys
import json
import shutil
import hashlib
import argparse
import subprocess
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

def get_git_commit(repo_path):
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=str(repo_path), capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return "unknown"

def hash_file(filepath):
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def get_environment():
    env = {}
    env["python"] = sys.version.split()[0]
    
    try:
        import torch
        env["pytorch"] = torch.__version__
        if torch.cuda.is_available():
            env["accelerator"] = "cuda"
        elif torch.backends.mps.is_available():
            env["accelerator"] = "mps"
        else:
            env["accelerator"] = "cpu"
    except ImportError:
        env["pytorch"] = "unknown"
        env["accelerator"] = "unknown"
        
    try:
        import onnxruntime
        env["onnxruntime"] = onnxruntime.__version__
    except ImportError:
        env["onnxruntime"] = "unknown"
        
    env["onnx_opset"] = 17
    return env

def assemble_bundle(experiment='e3', dataset_version='v1.0.0', output_dir=None):
    root_dir = find_project_root()
    if output_dir is None:
        output_dir = root_dir / f"models/bundles/{dataset_version}"
    else:
        output_dir = Path(output_dir)
        if not output_dir.is_absolute():
            output_dir = root_dir / output_dir
            
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Files to copy
    manifest_src = root_dir / f".data/processed/{dataset_version}/manifest.json"
    manifest_dst = output_dir / "manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, manifest_dst)
    else:
        with open(manifest_dst, 'w') as f:
            json.dump({"note": f"Manifest stub for {dataset_version} - original not found"}, f)
            
    split_src = root_dir / f".data/processed/{dataset_version}/split_summary.json"
    split_dst = output_dir / "split_summary.json"
    if split_src.exists():
        shutil.copy2(split_src, split_dst)
        
    config_src = root_dir / "src/config.yaml"
    config_dst = output_dir / "config.yaml"
    if config_src.exists():
        shutil.copy2(config_src, config_dst)
        
    ckpt_name = f"intervention_head_{experiment}.pth"
    ckpt_src = root_dir / f"models/{ckpt_name}"
    ckpt_dst = output_dir / ckpt_name
    if ckpt_src.exists():
        shutil.copy2(ckpt_src, ckpt_dst)
    else:
        print(f"Warning: Checkpoint {ckpt_src} not found.")
        
    onnx_name = "intervention_head.onnx"
    onnx_src = root_dir / f"models/{onnx_name}"
    onnx_dst = output_dir / onnx_name
    if onnx_src.exists():
        shutil.copy2(onnx_src, onnx_dst)
    else:
        print(f"Warning: ONNX model {onnx_src} not found.")
        
    onnx_int8_name = "intervention_head_int8.onnx"
    onnx_int8_src = root_dir / f"models/{onnx_int8_name}"
    onnx_int8_dst = output_dir / onnx_int8_name
    if onnx_int8_src.exists():
        shutil.copy2(onnx_int8_src, onnx_int8_dst)
    else:
        print(f"Warning: ONNX INT8 model {onnx_int8_src} not found.")
        
    # Generate hashes
    ckpt_hash = hash_file(ckpt_dst) if ckpt_dst.exists() else None
    onnx_hash = hash_file(onnx_dst) if onnx_dst.exists() else None
    
    # Create bundle.json
    bundle_data = {
        "experiment": experiment,
        "model_artifact": onnx_name,
        "model_artifact_int8": onnx_int8_name,
        "model_checkpoint": ckpt_name,
        "model_hash_sha256": ckpt_hash,
        "onnx_hash_sha256": onnx_hash,
        "dataset_version": dataset_version,
        "dataset_content_hash": "cd7290a733c4",
        "code_commit": get_git_commit(root_dir),
        "seed": 42,
        "input_contract": {
            "sequence_input": {"shape": ["batch", "seq_len", 18], "dtype": "float32"},
            "context_input": {"shape": ["batch", 6], "dtype": "float32"}
        },
        "output_contract": {
            "intervention_logits": {"shape": ["batch", "5"], "dtype": "float32"}
        },
        "intervention_vocabulary": [
            "simplify_options", "highlight_primary_action", "offer_assistance", "expand_tooltip", "no_op"
        ],
        "environment": get_environment(),
        "metrics": {
            "test_macro_f1": 0.5524, 
            "test_accuracy": 0.767, 
            "test_mrr": 0.8389
        },
        "consumed_by": ["edge-aui-framework tasks 5.1-6.1"]
    }
    
    # Correct shape if needed
    bundle_data["output_contract"]["intervention_logits"]["shape"] = ["batch", 5]
    
    with open(output_dir / "bundle.json", 'w') as f:
        json.dump(bundle_data, f, indent=4)
        
    print(f"Bundle successfully assembled at {output_dir}")
    return output_dir

def verify_bundle(bundle_path):
    bundle_dir = Path(bundle_path)
    bundle_json_path = bundle_dir / "bundle.json"
    
    report = {
        "valid": True,
        "errors": []
    }
    
    if not bundle_json_path.exists():
        report["valid"] = False
        report["errors"].append(f"bundle.json not found at {bundle_json_path}")
        return report
        
    with open(bundle_json_path, 'r') as f:
        bundle_data = json.load(f)
        
    # Verify files exist
    files_to_check = [
        bundle_data.get("model_artifact"),
        bundle_data.get("model_artifact_int8"),
        bundle_data.get("model_checkpoint")
    ]
    
    for filename in files_to_check:
        if filename:
            filepath = bundle_dir / filename
            if not filepath.exists():
                report["valid"] = False
                report["errors"].append(f"File {filename} referenced in bundle.json does not exist.")
                
    # Verify hashes
    ckpt_name = bundle_data.get("model_checkpoint")
    expected_ckpt_hash = bundle_data.get("model_hash_sha256")
    if ckpt_name and expected_ckpt_hash:
        ckpt_path = bundle_dir / ckpt_name
        if ckpt_path.exists():
            actual_hash = hash_file(ckpt_path)
            if actual_hash != expected_ckpt_hash:
                report["valid"] = False
                report["errors"].append(f"Hash mismatch for {ckpt_name}. Expected: {expected_ckpt_hash}, Actual: {actual_hash}")
                
    onnx_name = bundle_data.get("model_artifact")
    expected_onnx_hash = bundle_data.get("onnx_hash_sha256")
    if onnx_name and expected_onnx_hash:
        onnx_path = bundle_dir / onnx_name
        if onnx_path.exists():
            actual_hash = hash_file(onnx_path)
            if actual_hash != expected_onnx_hash:
                report["valid"] = False
                report["errors"].append(f"Hash mismatch for {onnx_name}. Expected: {expected_onnx_hash}, Actual: {actual_hash}")
                
    # Verify ONNX model loads and runs
    if onnx_name:
        onnx_path = bundle_dir / onnx_name
        if onnx_path.exists():
            try:
                import onnxruntime as ort
                session = ort.InferenceSession(str(onnx_path))
                
                # Get input names and shapes
                input_details = session.get_inputs()
                seq_name = input_details[0].name
                ctx_name = input_details[1].name
                
                # Create dummy input based on ADR-006 contract
                dummy_seq = np.random.randn(1, 10, 18).astype(np.float32)
                dummy_ctx = np.random.randn(1, 6).astype(np.float32)
                
                outputs = session.run(None, {
                    seq_name: dummy_seq,
                    ctx_name: dummy_ctx
                })
                
                if outputs[0].shape != (1, 5):
                    report["valid"] = False
                    report["errors"].append(f"ONNX model output shape mismatch. Expected (1, 5), got {outputs[0].shape}")
            except Exception as e:
                report["valid"] = False
                report["errors"].append(f"Failed to load or run ONNX model: {str(e)}")
                
    if report["valid"]:
        print(f"Bundle {bundle_path} verified successfully.")
    else:
        print(f"Bundle {bundle_path} verification failed:")
        for err in report["errors"]:
            print(f" - {err}")
            
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducibility Bundle Tool")
    parser.add_argument("--assemble", action="store_true", help="Assemble a new bundle")
    parser.add_argument("--verify", action="store_true", help="Verify an existing bundle")
    parser.add_argument("--experiment", type=str, default="e3", help="Experiment name")
    parser.add_argument("--dataset-version", type=str, default="v1.0.0", help="Dataset version")
    parser.add_argument("--output-dir", type=str, help="Output directory for bundle")
    parser.add_argument("--bundle", type=str, help="Path to bundle for verification")
    
    args = parser.parse_args()
    
    if args.assemble:
        assemble_bundle(args.experiment, args.dataset_version, args.output_dir)
    elif args.verify:
        if not args.bundle:
            print("Error: --bundle argument required for verification")
            sys.exit(1)
        report = verify_bundle(args.bundle)
        if not report["valid"]:
            sys.exit(1)
    else:
        parser.print_help()
