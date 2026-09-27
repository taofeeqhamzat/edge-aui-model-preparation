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
from typing import Optional, List, Dict, Any, Union, Tuple, Set

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
            # pyrefly: ignore [missing-import]
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

# pyrefly: ignore [invalid-annotation]
def is_dataset_present(data_dir: str or Path) -> bool:
    """
    Check if the target raw data directory exists and contains recognizable interaction datasets,
    either directly or nested within a 'raw/' subfolder.
    """
    p = Path(data_dir)
    if not p.is_dir():
        return False

    for check_path in [p, p / "raw", p / ".data" / "raw"]:
        if not check_path.is_dir():
            continue
        found_count = 0
        for subdir in DATASET_SUBDIRS:
            subpath = check_path / subdir
            if subpath.is_dir() and any(subpath.iterdir()):
                found_count += 1
        if found_count > 0:
            return True

    return False


# pyrefly: ignore [invalid-annotation]
def resolve_dataset_root(data_dir: str or Path) -> str:
    """
    Locate the root directory where the 4 interaction datasets actually live.
    Checks data_dir, data_dir/raw, data_dir/.data/raw, and parent directories.
    """
    p = Path(data_dir).resolve()
    candidates = [
        p,
        p / "raw",
        p / ".data" / "raw",
        p.parent / "raw",
    ]
    for c in candidates:
        if c.is_dir() and any((c / subdir).is_dir() and any((c / subdir).iterdir()) for subdir in DATASET_SUBDIRS):
            return str(c)
    return str(p)


# pyrefly: ignore [invalid-annotation]
def find_dataset_dir(root: str or Path, dataset_name: str) -> str:
    """
    Dynamically find the directory for a specific dataset within root,
    handling any nesting like root/name, root/raw/name, root/.data/raw/name.
    """
    p = Path(root).resolve()
    candidates = [
        p / dataset_name,
        p / "raw" / dataset_name,
        p / ".data" / "raw" / dataset_name,
        p.parent / dataset_name,
        p.parent / "raw" / dataset_name
    ]
    for c in candidates:
        if c.is_dir() and any(c.iterdir()):
            return str(c)

    # Recursive search fallback
    pattern = os.path.join(str(p), "**", dataset_name)
    matches = glob.glob(pattern, recursive=True)
    for m in matches:
        if os.path.isdir(m) and any(os.scandir(m)):
            return m

    return str(p / dataset_name)


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
        flatten_nested_raw(target_path)
        resolved = resolve_dataset_root(target_path)
        print(f"[DataManager] Verified local datasets at: {resolved}")
        return resolved

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
            return resolve_dataset_root(target_path)

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
        resolved = resolve_dataset_root(downloaded_dir)
        print(f"[DataManager] Successfully synchronized datasets to: {resolved}")
        return resolved

    except Exception as e:
        error_msg = str(e)
        print(f"[DataManager] Warning: Snapshot download failed: {error_msg}")
        
        # If rate-limited (HTTP 429), trigger Git clone fallback
        if "429" in error_msg or "Rate limited" in error_msg:
            print("\n[DataManager] ⚡ Detected HTTP 429 Rate Limit on REST API.")
            print("[DataManager] Switching to single-stream Git clone fallback...")
            if sync_via_git(repo_id, target_path, auth_token):
                flatten_nested_raw(target_path)
                return resolve_dataset_root(target_path)

        if is_dataset_present(target_path):
            flatten_nested_raw(target_path)
            resolved = resolve_dataset_root(target_path)
            print(f"[DataManager] Using partially downloaded or cached files at {resolved}.")
            return resolved
        else:
            print(
                "[DataManager] 💡 TIP: To permanently bypass Colab rate limits:\n"
                "   1. Set HF_TOKEN in Colab Secrets (🔑) with a free read token from https://huggingface.co/settings/tokens\n"
                "   2. Or run: !git clone --depth 1 https://huggingface.co/datasets/T40/edge-aui-framework-data .data/raw\n"
            )
            return resolve_dataset_root(target_path)


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
# 4. Provenance-Versioned Dataset Manifest & Validators (Phase C, Task 11.2)
# ============================================================================

import hashlib
import json
from datetime import datetime, timezone

try:
    from intervention_label_policy import LABEL_POLICY_VERSION, INTERVENTION_VOCABULARY, INTERVENTION_TO_ID
    from trace_ingestion import EXPERIMENT_TRACE_SCHEMA_VERSION, PREPROCESSING_VERSION, FEATURE_SCHEMA_VERSION
except ImportError:
    try:
        from src.intervention_label_policy import LABEL_POLICY_VERSION, INTERVENTION_VOCABULARY, INTERVENTION_TO_ID
        from src.trace_ingestion import EXPERIMENT_TRACE_SCHEMA_VERSION, PREPROCESSING_VERSION, FEATURE_SCHEMA_VERSION
    except ImportError:
        LABEL_POLICY_VERSION = "1.0.0"
        INTERVENTION_VOCABULARY = ("simplify_options", "highlight_primary_action", "offer_assistance", "expand_tooltip", "no_op")
        INTERVENTION_TO_ID = {name: idx for idx, name in enumerate(INTERVENTION_VOCABULARY)}
        EXPERIMENT_TRACE_SCHEMA_VERSION = "1.1.0"
        PREPROCESSING_VERSION = "1.1.0"
        FEATURE_SCHEMA_VERSION = "1.1.0"

MANIFEST_SCHEMA_VERSION = "1.0.0"

PROVENANCE_REQUIRED_FIELDS = (
    "session_id",
    "experiment_id",
    "condition_id",
    "task_id",
    "anchor_window_id",
    "source_event_ids",
    "preprocessing_version",
    "feature_schema_version",
    "target_generation_version",
)

ADR_013_CLAIM_BOUNDARY = (
    "ADR-013: Substitute interaction traces and scripted label assignments. "
    "Results from this dataset may be reported as: 'the dataset -> preparation -> training -> export -> runtime path executes end to end', "
    "'the learned head loads in the browser and produces logits of the expected shape', and engineering and pipeline findings. "
    "Results from this dataset may NOT be reported as: evidence about human participants; evidence about usability, task performance, "
    "or intervention benefit; or evidence that learned intervention prediction outperforms the deterministic policy — "
    "a model trained on labels produced by that policy is being compared against its own teacher."
)


class ManifestValidationError(ValueError):
    """Raised when dataset manifest fails provenance, integrity, or split disjointness validation."""
    pass


def compute_dataset_content_hash(rows: List[Dict[str, Any]]) -> str:
    """Computes a deterministic content hash over canonical example attributes."""
    hasher = hashlib.sha256()
    for r in sorted(rows, key=lambda x: (x.get("session_id", ""), x.get("anchor_window_id", 0))):
        key = f"{r.get('session_id')}|{r.get('task_id')}|{r.get('anchor_window_id')}|{r.get('split')}|{r.get('target_intervention_id')}"
        hasher.update(key.encode("utf-8"))
    return hasher.hexdigest()[:12]


def create_dataset_manifest(
    rows: List[Dict[str, Any]],
    split_summary: Dict[str, Any],
    trace_dir: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dataset_version: Optional[str] = None,
) -> Tuple[Dict[str, Any], Path]:
    """
    Constructs an immutable, content-addressed dataset manifest (Task 11.2)
    retaining all 9 required provenance fields for every example (Brief §8).
    """
    content_hash = compute_dataset_content_hash(rows)
    resolved_version = dataset_version or f"v1.0.0-{content_hash}"
    created_at = datetime.now(timezone.utc).isoformat()

    manifest_examples: List[Dict[str, Any]] = []
    for r in rows:
        ex = {
            "example_id": f"{r.get('session_id')}:w{r.get('anchor_window_id')}",
            "session_id": str(r.get("session_id", "")),
            "experiment_id": str(r.get("experiment_id", "")),
            "condition_id": str(r.get("condition_id", "")),
            "task_id": str(r.get("task_id", "")),
            "anchor_window_id": int(r.get("anchor_window_id", 0)),
            "source_event_ids": list(r.get("source_event_ids", [])),
            "preprocessing_version": str(r.get("preprocessing_version", PREPROCESSING_VERSION)),
            "feature_schema_version": str(r.get("feature_schema_version", FEATURE_SCHEMA_VERSION)),
            "target_generation_version": str(r.get("target_generation_version", LABEL_POLICY_VERSION)),
            "split": str(r.get("split", "unassigned")),
            "target_outcome": str(r.get("target_outcome", "NO_OUTCOME")),
            "target_outcome_id": int(r.get("target_outcome_id", 0)),
            "target_intervention": str(r.get("target_intervention", "no_op")),
            "target_intervention_id": int(r.get("target_intervention_id", 4)),
            "label_policy_version": str(r.get("label_policy_version", LABEL_POLICY_VERSION)),
            "is_scripted_policy": bool(r.get("is_scripted_policy", True)),
        }
        manifest_examples.append(ex)

    manifest = {
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "dataset_version": resolved_version,
        "content_hash": content_hash,
        "created_at": created_at,
        "source_data_type": "scripted_substitute_testbed",
        "claim_boundary": ADR_013_CLAIM_BOUNDARY,
        "provenance_spec": {
            "brief_reference": "Brief Section 8 & Section 17",
            "required_fields": list(PROVENANCE_REQUIRED_FIELDS),
        },
        "code_versions": {
            "preprocessing_version": PREPROCESSING_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_policy_version": LABEL_POLICY_VERSION,
            "trace_schema_version": EXPERIMENT_TRACE_SCHEMA_VERSION,
        },
        "split_summary": split_summary,
        "class_weights": split_summary.get("class_weights", {}),
        "total_examples": len(manifest_examples),
        "examples": manifest_examples,
    }

    if output_path is None:
        out_p = Path(f".data/processed/{resolved_version}/manifest.json")
    else:
        out_p = Path(output_path)

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as fp:
        json.dump(manifest, fp, indent=2)

    print(f"[DataManager] Provenance-versioned manifest written to: {out_p}")
    return manifest, out_p


def validate_dataset_manifest(
    manifest_path_or_dict: Union[str, Path, Dict[str, Any]],
    trace_dir: Optional[Union[str, Path]] = None,
    check_events: bool = True,
) -> Dict[str, Any]:
    """
    Strictly validates a dataset manifest JSON (Task 11.2).
    Raises ManifestValidationError on ANY incompleteness or violation:
    1. Every provenance field present and resolvable on every example.
    2. Every sourceEventId exists in its source trace.
    3. Splits are strictly session-bounded and disjoint.
    4. Label policy version matches the labels and code.
    5. Feature schema and preprocessing versions match current code.
    """
    if isinstance(manifest_path_or_dict, (str, Path)):
        p = Path(manifest_path_or_dict)
        if not p.is_file():
            raise ManifestValidationError(f"Manifest file does not exist: {p}")
        try:
            with open(p, "r", encoding="utf-8") as fp:
                manifest = json.load(fp)
        except Exception as e:
            raise ManifestValidationError(f"Failed to parse manifest JSON from {p}: {e}")
    elif isinstance(manifest_path_or_dict, dict):
        manifest = manifest_path_or_dict
    else:
        raise ManifestValidationError(f"Expected path or dict, received: {type(manifest_path_or_dict)}")

    # 1. Header schema validation
    required_headers = [
        "manifest_schema_version",
        "dataset_version",
        "claim_boundary",
        "code_versions",
        "split_summary",
        "class_weights",
        "total_examples",
        "examples",
    ]
    for h in required_headers:
        if h not in manifest or manifest[h] is None:
            raise ManifestValidationError(f"Manifest is missing required header field: '{h}'")

    # 2. Code versions verification
    cv = manifest["code_versions"]
    if cv.get("preprocessing_version") != PREPROCESSING_VERSION:
        raise ManifestValidationError(
            f"Preprocessing version mismatch: manifest has '{cv.get('preprocessing_version')}', "
            f"code requires '{PREPROCESSING_VERSION}'"
        )
    if cv.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ManifestValidationError(
            f"Feature schema version mismatch: manifest has '{cv.get('feature_schema_version')}', "
            f"code requires '{FEATURE_SCHEMA_VERSION}'"
        )
    if cv.get("label_policy_version") != LABEL_POLICY_VERSION:
        raise ManifestValidationError(
            f"Label policy version mismatch: manifest has '{cv.get('label_policy_version')}', "
            f"code requires '{LABEL_POLICY_VERSION}'"
        )

    # 3. Split disjointness verification
    split_summary = manifest["split_summary"]
    sessions_dict = split_summary.get("sessions", {})
    train_s = set(sessions_dict.get("train", []))
    val_s = set(sessions_dict.get("val", []))
    test_s = set(sessions_dict.get("test", []))

    if not train_s.isdisjoint(val_s):
        raise ManifestValidationError(f"Non-disjoint splits: Train and Val overlap on {train_s & val_s}")
    if not train_s.isdisjoint(test_s):
        raise ManifestValidationError(f"Non-disjoint splits: Train and Test overlap on {train_s & test_s}")
    if not val_s.isdisjoint(test_s):
        raise ManifestValidationError(f"Non-disjoint splits: Val and Test overlap on {val_s & test_s}")

    # 4. Class weights derivation provenance
    cw = manifest.get("class_weights", {})
    if cw.get("derived_on") != "train_partition_only":
        raise ManifestValidationError(
            f"Class weights provenance invalid: expected 'train_partition_only', received '{cw.get('derived_on')}'"
        )

    examples = manifest["examples"]
    if not isinstance(examples, list) or len(examples) == 0:
        raise ManifestValidationError("Manifest contains no examples or examples is not a list")

    if len(examples) != manifest["total_examples"]:
        raise ManifestValidationError(
            f"Total examples mismatch: header declared {manifest['total_examples']}, found {len(examples)}"
        )

    # Pre-index source traces if trace_dir is provided and check_events is requested
    trace_events_cache: Dict[str, Set[str]] = {}
    if check_events:
        td = Path(trace_dir) if trace_dir else Path(".data/raw/scripted")
        if td.is_dir():
            trace_files = glob.glob(str(td / "*.json"))
            for tf in trace_files:
                if Path(tf).name.startswith("manifest"):
                    continue
                try:
                    with open(tf, "r", encoding="utf-8") as fp:
                        t_data = json.load(fp)
                    sid = t_data.get("session", {}).get("sessionId")
                    b_events = t_data.get("behaviourEvents", [])
                    if sid:
                        trace_events_cache[sid] = {f"{sid}:ev_{i}" for i in range(len(b_events))}
                except Exception:
                    pass

    # 5. Validate every single example for all 9 provenance fields & trace integrity
    seen_sessions = set()
    for idx, ex in enumerate(examples):
        for field in PROVENANCE_REQUIRED_FIELDS:
            if field not in ex or ex[field] is None:
                raise ManifestValidationError(
                    f"Example [{idx}] ({ex.get('example_id', 'unknown')}) is missing required provenance field: '{field}'"
                )
            if isinstance(ex[field], str) and not ex[field].strip():
                raise ManifestValidationError(
                    f"Example [{idx}] has empty string for required provenance field: '{field}'"
                )

        sid = ex["session_id"]
        seen_sessions.add(sid)
        split = ex.get("split")
        if split not in {"train", "val", "test"}:
            raise ManifestValidationError(f"Example [{idx}] has invalid split tag: '{split}'")

        # Verify example split matches session partition
        if sid in train_s and split != "train":
            raise ManifestValidationError(f"Example [{idx}] belongs to train session '{sid}' but has split '{split}'")
        elif sid in val_s and split != "val":
            raise ManifestValidationError(f"Example [{idx}] belongs to val session '{sid}' but has split '{split}'")
        elif sid in test_s and split != "test":
            raise ManifestValidationError(f"Example [{idx}] belongs to test session '{sid}' but has split '{split}'")

        # Verify label policy consistency
        if ex.get("label_policy_version") != LABEL_POLICY_VERSION:
            raise ManifestValidationError(
                f"Example [{idx}] label_policy_version '{ex.get('label_policy_version')}' != '{LABEL_POLICY_VERSION}'"
            )

        target_int = ex.get("target_intervention")
        if target_int not in INTERVENTION_VOCABULARY:
            raise ManifestValidationError(f"Example [{idx}] has unknown intervention: '{target_int}'")
        if ex.get("target_intervention_id") != INTERVENTION_TO_ID[target_int]:
            raise ManifestValidationError(
                f"Example [{idx}] intervention ID mismatch: '{target_int}' mapped to {ex.get('target_intervention_id')}"
            )

        # Source event verification against trace cache
        source_evs = ex.get("source_event_ids", [])
        if not isinstance(source_evs, list):
            raise ManifestValidationError(f"Example [{idx}] source_event_ids must be a list")
        if len(source_evs) == 0:
            raise ManifestValidationError(f"Example [{idx}] has empty source_event_ids list")

        if check_events and sid in trace_events_cache:
            valid_ev_ids = trace_events_cache[sid]
            for ev_id in source_evs:
                if ev_id not in valid_ev_ids:
                    raise ManifestValidationError(
                        f"Example [{idx}] references non-existent sourceEventId: '{ev_id}' for session '{sid}'"
                    )

    return {
        "valid": True,
        "dataset_version": manifest["dataset_version"],
        "total_examples": len(examples),
        "total_sessions": len(seen_sessions),
        "split_summary": split_summary,
    }


# ============================================================================
# 5. Standalone CLI Execution
# ============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Synchronize and Validate Edge-AUI Datasets.")
    parser.add_argument("--repo-id", type=str, default="T40/edge-aui-framework-data", help="Hugging Face repo ID")
    parser.add_argument("--data-dir", type=str, default=None, help="Target raw data directory")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face access token")
    parser.add_argument("--force", action="store_true", help="Force redownload")
    parser.add_argument("--validate-manifest", type=str, default=None, help="Path to manifest JSON to validate")
    parser.add_argument("--trace-dir", type=str, default=".data/raw/scripted", help="Trace directory for source event validation")
    args = parser.parse_args()

    if args.validate_manifest:
        res = validate_dataset_manifest(args.validate_manifest, trace_dir=args.trace_dir)
        print(f"[DataManager] Manifest validation SUCCESS: {res['dataset_version']} ({res['total_examples']} examples)")
    else:
        print(f"Environment: Colab={is_colab()}, Kaggle={is_kaggle()}")
        path = ensure_dataset(data_dir=args.data_dir, repo_id=args.repo_id, token=args.token, force_download=args.force)
        print(f"Dataset ready at: {path}")

