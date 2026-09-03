"""
data_manager.py
Hosted Dataset Manager & Environment Resolver for Edge-AUI Framework.
Handles dynamic synchronization from Hugging Face Hub (T40/edge-aui-framework-data),
local caching, environment auto-detection (Colab, Kaggle, Local), and token authentication.
"""

import os
import sys
import glob
from pathlib import Path
from typing import Optional, List, Dict, Any

# ============================================================================
# 1. Environment Detection & Token Resolution
# ============================================================================

def is_colab() -> bool:
    """Detect if running inside Google Colaboratory."""
    return "google.colab" in sys.modules or "COLAB_GPU" in os.environ

def is_kaggle() -> bool:
    """Detect if running inside Kaggle Kernels."""
    return "KAGGLE_KERNEL_RUN_TYPE" in os.environ

def get_hf_token(token: Optional[str] = None) -> Optional[str]:
    """
    Retrieve Hugging Face authentication token across environments:
    1. Explicit parameter
    2. HF_TOKEN environment variable
    3. Google Colab Secrets (userdata)
    4. None (fallback to public repository access)
    """
    if token:
        return token

    if "HF_TOKEN" in os.environ:
        return os.environ["HF_TOKEN"]

    if is_colab():
        try:
            from google.colab import userdata
            return userdata.get("HF_TOKEN")
        except Exception:
            pass

    return None

def find_project_root() -> Path:
    """
    Traverse upwards to locate project root (containing src/, requirements.txt, or .git).
    Falls back to current working directory if not located.
    """
    current = Path(os.getcwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").is_dir() or (parent / "requirements.txt").is_file() or (parent / ".git").is_dir():
            return parent
    return current


# ============================================================================
# 2. Local Dataset Verification
# ============================================================================

DATASET_SUBDIRS = [
    "continuous-kinematics-2020",
    "high-volume-trajectories-20226",
    "structural-hmi-sequences-2023",
    "client-side-action-paths-2021"
]

def is_dataset_present(data_dir: str or Path) -> bool:
    """
    Check if the target raw data directory exists and contains recognizable interaction datasets.
    """
    p = Path(data_dir)
    if not p.is_dir():
        return False

    # Check for presence of any of the expected datasets with non-empty files
    found_count = 0
    for subdir in DATASET_SUBDIRS:
        subpath = p / subdir
        if subpath.is_dir() and any(subpath.iterdir()):
            found_count += 1

    return found_count > 0


# ============================================================================
# 3. Hosted Synchronization via Hugging Face Hub
# ============================================================================

def ensure_dataset(
    data_dir: Optional[str] = None,
    repo_id: str = "T40/edge-aui-framework-data",
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    force_download: bool = False
) -> str:
    """
    Ensure the foundational raw interaction datasets are accessible locally.
    
    If data_dir contains the datasets, it returns the path immediately (0 network cost).
    If missing or empty (e.g., fresh Colab / Kaggle runtime), it automatically downloads
    and caches the datasets from the Hugging Face Hub repository.
    
    Args:
        data_dir: Local path to .data/raw directory (defaults to <project_root>/.data/raw).
        repo_id: Hugging Face dataset repository identifier.
        token: Optional HF authentication token.
        allow_patterns: Optional file glob patterns to selectively download subsets.
        force_download: If True, forces redownload even if local data exists.
        
    Returns:
        str: Absolute path to the verified local raw dataset directory.
    """
    if data_dir is None:
        root = find_project_root()
        target_path = root / ".data" / "raw"
    else:
        target_path = Path(data_dir).resolve()

    target_path.mkdir(parents=True, exist_ok=True)

    if not force_download and is_dataset_present(target_path):
        print(f"[DataManager] Verified local datasets at: {target_path}")
        return str(target_path)

    print(f"[DataManager] Datasets not found or update requested. Syncing from Hugging Face: '{repo_id}'...")
    auth_token = get_hf_token(token)

    try:
        from huggingface_hub import snapshot_download
        
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_path),
            allow_patterns=allow_patterns,
            token=auth_token,
            resume_download=True
        )
        print(f"[DataManager] Successfully synchronized datasets to: {downloaded_dir}")
        return str(downloaded_dir)

    except Exception as e:
        print(f"[DataManager] Warning: Automatic sync from Hugging Face failed: {e}")
        if is_dataset_present(target_path):
            print(f"[DataManager] Falling back to existing files at {target_path}.")
            return str(target_path)
        else:
            print(
                "[DataManager] Note: If the repository is private, please set your HF_TOKEN "
                "environment variable or use `huggingface_hub.login()`."
            )
            return str(target_path)


def load_hosted_dataset(
    repo_id: str = "T40/edge-aui-framework-data",
    subset: Optional[str] = None,
    split: str = "train",
    streaming: bool = False,
    token: Optional[str] = None
) -> Any:
    """
    Dynamically stream or load dataset records directly into memory using the
    Hugging Face `datasets` library, bypassing local disk storage.
    """
    try:
        from datasets import load_dataset
        auth_token = get_hf_token(token)
        print(f"[DataManager] Loading dataset '{repo_id}' (subset={subset}, streaming={streaming})...")
        ds = load_dataset(repo_id, name=subset, split=split, streaming=streaming, token=auth_token)
        return ds
    except Exception as e:
        print(f"[DataManager] Error streaming from Hugging Face: {e}")
        raise


# ============================================================================
# 4. Standalone CLI Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synchronize Edge-AUI interaction datasets from Hugging Face Hub.")
    parser.add_argument("--repo-id", type=str, default="T40/edge-aui-framework-data", help="Hugging Face repo ID")
    parser.add_argument("--data-dir", type=str, default=None, help="Target raw data directory")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    args = parser.parse_args()

    print(f"Environment: Colab={is_colab()}, Kaggle={is_kaggle()}")
    path = ensure_dataset(data_dir=args.data_dir, repo_id=args.repo_id, token=args.token, force_download=args.force)
    print(f"Dataset ready at: {path}")
