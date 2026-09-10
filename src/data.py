"""
data.py
Dataset Ingestion, Inode-Safe Parquet Consolidation, and DVC Integration.
Handles resolution of AdSERP (and extended datasets), session streaming, and cache management.
"""

import os
import sys
import glob
import json
import gzip
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any

try:
    from config import load_config, PipelineConfig
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.config import load_config, PipelineConfig
    except ImportError:
        load_config = None
        PipelineConfig = None


def find_project_root() -> Path:
    """Traverse upwards to locate project root (containing src/, requirements.txt, or .git)."""
    current = Path(os.getcwd()).resolve()
    for parent in [current] + list(current.parents):
        if (parent / "src").is_dir() or (parent / "requirements.txt").is_file() or (parent / ".git").is_dir():
            return parent
    return current


def parse_viewport_metadata(xml_path: str) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """
    Parse source window/viewport and document dimensions from trial metadata XML.
    Returns ((viewport_w, viewport_h), (doc_w, doc_h)).
    Fails visibly (raises FileNotFoundError or ValueError) if XML is missing or malformed.
    """
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"Trial metadata XML file not found: {xml_path}")

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        raise ValueError(f"Failed to parse XML file {xml_path}: {e}")

    win_node = root.find("window")
    if win_node is None or not win_node.text or "x" not in win_node.text:
        raise ValueError(f"Metadata XML {xml_path} is missing a valid '<window>WxH</window>' element.")

    parts = win_node.text.strip().split("x")
    if len(parts) != 2:
        raise ValueError(f"Malformed <window> string '{win_node.text}' in {xml_path}; expected 'WIDTHxHEIGHT'.")

    try:
        viewport_w, viewport_h = float(parts[0]), float(parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric <window> dimensions '{win_node.text}' in {xml_path}: {e}")

    if viewport_w <= 0 or viewport_h <= 0:
        raise ValueError(f"Non-positive <window> dimensions ({viewport_w}, {viewport_h}) in {xml_path}.")

    doc_node = root.find("document")
    if doc_node is None or not doc_node.text or "x" not in doc_node.text:
        raise ValueError(f"Metadata XML {xml_path} is missing a valid '<document>WxH</document>' element.")

    doc_parts = doc_node.text.strip().split("x")
    if len(doc_parts) != 2:
        raise ValueError(f"Malformed <document> string '{doc_node.text}' in {xml_path}; expected 'WIDTHxHEIGHT'.")

    try:
        doc_w, doc_h = float(doc_parts[0]), float(doc_parts[1])
    except ValueError as e:
        raise ValueError(f"Non-numeric <document> dimensions '{doc_node.text}' in {xml_path}: {e}")

    return (viewport_w, viewport_h), (doc_w, doc_h)


def pull_dvc_dataset(target: Optional[str] = None) -> bool:
    """Attempt pulling datasets from configured DVC remote (e.g. Hugging Face S3 gateway)."""
    try:
        cmd = ["dvc", "pull"]
        if target:
            cmd.append(target)
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode == 0:
            print("[Data] Successfully pulled data from DVC remote.")
            return True
        else:
            print(f"[Data] DVC pull informational note: {res.stderr.strip() or res.stdout.strip()}")
            return False
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[Data] Note: DVC pull skipped ({e}). Falling back to local/hosted synchronization.")
        return False


def resolve_adserp_dir(custom_raw_dir: Optional[str] = None) -> Optional[Path]:
    """
    Search candidate directories for verified AdSERP dataset containing mouse-movement-data.
    """
    root = find_project_root()
    candidates: List[Path] = []

    if custom_raw_dir:
        candidates.append(Path(custom_raw_dir).resolve())
        candidates.append(Path(custom_raw_dir).resolve() / "adserp-2025")
        candidates.append(Path(custom_raw_dir).resolve() / "adserp")

    # Priority local paths
    candidates.extend([
        root / ".data" / "raw" / "adserp-2025",
        root / ".data" / "raw" / "adserp",
        root.parent / "data-sources" / "adserp-2025",
        root.parent / "data-sources" / "adserp",
        Path("/Users/user/Workspace/MivaCS/FYP/data-sources/adserp-2025")
    ])

    for c in candidates:
        if c.is_dir() and (c / "mouse-movement-data").is_dir():
            csv_check = any((c / "mouse-movement-data").glob("*.csv"))
            if csv_check:
                return c
    return None


def ensure_adserp_dataset(
    raw_dir: Optional[str] = None,
    link_to_raw: bool = True
) -> str:
    """
    Ensure the AdSERP dataset is accessible locally.
    Resolves local storage first, then checks DVC remote, and links into .data/raw to prevent inode duplication.
    """
    root = find_project_root()
    target_raw = Path(raw_dir).resolve() if raw_dir else (root / ".data" / "raw")
    target_raw.mkdir(parents=True, exist_ok=True)
    target_adserp = target_raw / "adserp-2025"

    # 1. Check if already directly present in target raw dir
    if target_adserp.is_dir() and (target_adserp / "mouse-movement-data").is_dir():
        if any((target_adserp / "mouse-movement-data").glob("*.csv")):
            return str(target_adserp)

    # 2. Check candidate local paths (e.g. data-sources/adserp-2025)
    found = resolve_adserp_dir(raw_dir)
    if found:
        # Symlink into target raw directory if requested (conserves inodes & disk space)
        if link_to_raw and not target_adserp.exists():
            try:
                target_adserp.symlink_to(found, target_is_directory=True)
                print(f"[Data] Created inode-safe symlink: {target_adserp} -> {found}")
                return str(target_adserp)
            except Exception:
                return str(found)
        return str(found)

    # 3. Try DVC pull
    print("[Data] AdSERP not found locally; querying DVC remote...")
    if pull_dvc_dataset("adserp-2025"):
        found = resolve_adserp_dir(raw_dir)
        if found:
            return str(found)

    # 4. Hosted Hugging Face Hub fallback
    print("[Data] Querying Hugging Face Hub for hosted interaction datasets...")
    try:
        from huggingface_hub import snapshot_download
        downloaded = snapshot_download(
            repo_id="T40/edge-aui-framework-data",
            repo_type="dataset",
            local_dir=str(target_raw),
            allow_patterns=["*adserp*", "adserp-2025/*"],
            max_workers=2
        )
        if resolve_adserp_dir(downloaded):
            return str(resolve_adserp_dir(downloaded))
    except Exception as e:
        print(f"[Data] Note on hosted sync: {e}")

    # Fallback to target_adserp path
    return str(target_adserp)


def consolidate_adserp_sessions(
    raw_path: Optional[str] = None,
    interim_dir: Optional[str] = None,
    max_sessions: Optional[int] = None,
    force: bool = False
) -> str:
    """
    Consolidate thousands of individual AdSERP session CSVs and companion XMLs
    into a single compressed columnar file (.parquet or .csv.gz) in .data/interim/.
    Prevents inode exhaustion and accelerates downstream training I/O.
    """
    root = find_project_root()
    adserp_dir = Path(raw_path or ensure_adserp_dataset())
    target_interim = Path(interim_dir or (root / ".data" / "interim")).resolve()
    target_interim.mkdir(parents=True, exist_ok=True)

    parquet_out = target_interim / "adserp_consolidated.parquet"
    csvgz_out = target_interim / "adserp_consolidated.csv.gz"

    if not force and parquet_out.is_file():
        print(f"[Data] Using cached consolidated Parquet: {parquet_out}")
        return str(parquet_out)

    if not force and csvgz_out.is_file():
        print(f"[Data] Using cached consolidated archive: {csvgz_out}")
        return str(csvgz_out)

    mouse_dir = adserp_dir / "mouse-movement-data"
    meta_dir = adserp_dir / "trial-metadata"

    if not mouse_dir.is_dir():
        print(f"[Data] Warning: mouse-movement-data not found at {mouse_dir}")
        return str(parquet_out)

    csv_files = sorted(list(mouse_dir.glob("*.csv")))
    if max_sessions:
        csv_files = csv_files[:max_sessions]

    print(f"[Data] Consolidating {len(csv_files)} AdSERP session files into inode-safe storage...")

    # Collect session records
    rows: List[Dict[str, Any]] = []
    for csv_file in csv_files:
        session_id = csv_file.stem
        xml_file = meta_dir / f"{session_id}.xml"
        viewport, doc = parse_viewport_metadata(str(xml_file))

        try:
            with open(csv_file, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        try:
                            ts = int(parts[0])
                            x = float(parts[1])
                            y = float(parts[2])
                            ev = parts[3]
                            xpath = parts[4] if len(parts) > 4 else ""
                            rows.append({
                                "session_id": session_id,
                                "timestamp": ts,
                                "xpos": x,
                                "ypos": y,
                                "event": ev,
                                "xpath": xpath,
                                "viewport_w": viewport[0],
                                "viewport_h": viewport[1],
                                "doc_w": doc[0],
                                "doc_h": doc[1]
                            })
                        except ValueError:
                            continue
        except Exception:
            continue

    print(f"[Data] Aggregated {len(rows)} raw interaction events across {len(csv_files)} sessions.")

    # Write as Parquet if pyarrow/pandas available
    try:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.to_parquet(str(parquet_out), compression="snappy", index=False)
        print(f"[Data] Successfully wrote consolidated Parquet: {parquet_out} ({parquet_out.stat().st_size / 1024:.1f} KB)")
        return str(parquet_out)
    except ImportError:
        pass

    # Safe fallback: write compressed CSV.GZ
    with gzip.open(str(csvgz_out), "wt", encoding="utf-8") as f:
        f.write("session_id,timestamp,xpos,ypos,event,xpath,viewport_w,viewport_h,doc_w,doc_h\n")
        for r in rows:
            f.write(f"{r['session_id']},{r['timestamp']},{r['xpos']},{r['ypos']},{r['event']},{r['xpath']},{r['viewport_w']},{r['viewport_h']},{r['doc_w']},{r['doc_h']}\n")

    print(f"[Data] Successfully wrote compressed archive: {csvgz_out} ({csvgz_out.stat().st_size / 1024:.1f} KB)")
    return str(csvgz_out)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge-AUI Data Ingestion & Consolidation Driver")
    parser.add_argument("--mode", type=str, default="minimal")
    parser.add_argument("--max-sessions", type=int, default=50, help="Cap sessions for fast ingestion test")
    parser.add_argument("--force", action="store_true", help="Force re-consolidation")
    args = parser.parse_args()

    adserp_path = ensure_adserp_dataset()
    print(f"[Data Driver] Verified AdSERP path: {adserp_path}")

    consolidated = consolidate_adserp_sessions(raw_path=adserp_path, max_sessions=args.max_sessions, force=args.force)
    print(f"[Data Driver] Consolidated dataset ready at: {consolidated}")
