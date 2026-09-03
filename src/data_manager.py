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

def sync_via_git(
    repo_id: str,
    target_path: Path,
    token: Optional[str] = None
) -> bool:
    """
    Synchronize repository using Git shallow clone.
    Transfers files in a single packfile stream via Git Smart HTTP,
    completely avoiding individual HTTP GET requests that trigger HTTP 429 Rate Limits.
    """
    import subprocess
    import shutil

    auth_token = get_hf_token(token)
    if auth_token:
        git_url = f"https://oauth2:{auth_token}@huggingface.co/datasets/{repo_id}"
    else:
        git_url = f"https://huggingface.co/datasets/{repo_id}"

    temp_clone_dir = target_path.parent / ".hf_git_clone_tmp"
    if temp_clone_dir.exists():
        shutil.rmtree(temp_clone_dir, ignore_errors=True)

    print(f"[DataManager] Cloning via Git from {repo_id} (bypasses REST API 429 rate limits)...")
    try:
        cmd = ["git", "clone", "--depth", "1", git_url, str(temp_clone_dir)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            print(f"[DataManager] Git clone encountered an error: {proc.stderr.strip()}")
            return False

        # Move files to target_path, flattening nested 'raw' directory if present
        source_dir = temp_clone_dir / "raw" if (temp_clone_dir / "raw").is_dir() else temp_clone_dir
        target_path.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.name == ".git":
                continue
            dest = target_path / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest, ignore_errors=True)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))

        shutil.rmtree(temp_clone_dir, ignore_errors=True)
        print(f"[DataManager] Successfully synchronized dataset via Git to: {target_path}")
        return True
    except Exception as e:
        print(f"[DataManager] Git clone fallback failed: {e}")
        if temp_clone_dir.exists():
            shutil.rmtree(temp_clone_dir, ignore_errors=True)
        return False


def flatten_nested_raw(target_path: Path):
    """If HF Hub downloads files under a nested 'raw/' subfolder, move them to target_path root."""
    import shutil
    nested_raw = target_path / "raw"
    if nested_raw.is_dir():
        for item in nested_raw.iterdir():
            dest = target_path / item.name
            if not dest.exists():
                shutil.move(str(item), str(dest))
        try:
            nested_raw.rmdir()
        except Exception:
            pass


def ensure_dataset(
    data_dir: Optional[str] = None,
    repo_id: str = "T40/edge-aui-framework-data",
    token: Optional[str] = None,
    allow_patterns: Optional[List[str]] = None,
    force_download: bool = False,
    method: str = "auto",
    max_workers: int = 2
) -> str:
    """
    Ensure the foundational raw interaction datasets are accessible locally.
    
    Handles Hugging Face API rate limits on Google Colab / shared IPs by:
    1. Checking local cache first (0 network cost).
    2. Throttling concurrent snapshot requests (max_workers=2) to avoid burst 429s.
    3. Auto-fallback to Git stream clone if HTTP 429 is encountered.
    4. Automatically flattening nested 'raw/' directory structures from HF Hub.
    
    Args:
        data_dir: Local path to raw data directory (defaults to <project_root>/.data/raw).
        repo_id: Hugging Face dataset repository identifier.
        token: Optional HF authentication token.
        allow_patterns: Optional file glob patterns to selectively download subsets.
        force_download: If True, forces redownload even if local data exists.
        method: Download method ('auto', 'snapshot', or 'git').
        max_workers: Concurrency limit for snapshot download (default 2 to prevent 429s).
        
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
        flatten_nested_raw(target_path)
        return str(target_path)

    auth_token = get_hf_token(token)
    if auth_token is None and (is_colab() or is_kaggle()):
        print(
            "\n[DataManager] ⚠️ WARNING: No Hugging Face token detected in cloud environment!\n"
            "   Anonymous requests from Google Colab shared IPs frequently hit HTTP 429 Rate Limits.\n"
            "   👉 Recommended: Add HF_TOKEN to Google Colab Secrets (🔑) or provide token='hf_...'\n"
        )

    # Strategy 1: Git-based clone if explicitly requested
    if method == "git":
        if sync_via_git(repo_id, target_path, auth_token):
            flatten_nested_raw(target_path)
            return str(target_path)

    # Strategy 2: Throttled snapshot download with 429 fallback
    print(f"[DataManager] Syncing from Hugging Face Hub: '{repo_id}' (workers={max_workers})...")
    try:
        from huggingface_hub import snapshot_download
        
        # Omit deprecated resume_download argument; limit max_workers to prevent HTTP 429
        downloaded_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(target_path),
            allow_patterns=allow_patterns,
            token=auth_token,
            max_workers=max_workers
        )
        flatten_nested_raw(Path(downloaded_dir))
        print(f"[DataManager] Successfully synchronized datasets to: {downloaded_dir}")
        return str(downloaded_dir)

    except Exception as e:
        error_msg = str(e)
        print(f"[DataManager] Warning: Snapshot download failed: {error_msg}")
        
        # If rate-limited (HTTP 429), trigger Git clone fallback
        if "429" in error_msg or "Rate limited" in error_msg:
            print("\n[DataManager] ⚡ Detected HTTP 429 Rate Limit on REST API.")
            print("[DataManager] Switching to single-stream Git clone fallback...")
            if sync_via_git(repo_id, target_path, auth_token):
                flatten_nested_raw(target_path)
                return str(target_path)

        if is_dataset_present(target_path):
            print(f"[DataManager] Using partially downloaded or cached files at {target_path}.")
            flatten_nested_raw(target_path)
            return str(target_path)
        else:
            print(
                "[DataManager] 💡 TIP: To permanently bypass Colab rate limits:\n"
                "   1. Set HF_TOKEN in Colab Secrets (🔑) with a free read token from https://huggingface.co/settings/tokens\n"
                "   2. Or run: !git clone --depth 1 https://huggingface.co/datasets/T40/edge-aui-framework-data .data/raw\n"
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
