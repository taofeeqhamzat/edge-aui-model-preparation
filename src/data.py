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
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    pa = None  # type: ignore
    pq = None  # type: ignore
    PYARROW_AVAILABLE = False

try:
    from config import load_config, PipelineConfig
except ImportError:
    try:
        # pyrefly: ignore [missing-import]
        from src.config import load_config, PipelineConfig
    except ImportError:
        load_config = None
        PipelineConfig = None

# Cross-dataset Canonical Event Schema
if PYARROW_AVAILABLE:
    CANONICAL_EVENT_SCHEMA = pa.schema([
        ("dataset_id", pa.string()),
        ("session_id", pa.string()),
        ("user_id", pa.string()),          # Nullable
        ("trial_id", pa.string()),         # Nullable
        ("timestamp_ms", pa.int64()),
        ("event_type", pa.string()),
        ("x", pa.float32()),               # Nullable
        ("y", pa.float32()),               # Nullable
        ("x_norm", pa.float32()),          # Nullable [0, 1]
        ("y_norm", pa.float32()),          # Nullable [0, 1]
        ("viewport_width", pa.float32()),  # Nullable
        ("viewport_height", pa.float32()), # Nullable
        ("document_width", pa.float32()),  # Nullable
        ("document_height", pa.float32()), # Nullable
        ("target_id", pa.string()),        # Nullable (XPath or DOM ID)
        ("source_event_id", pa.string()),  # Nullable
        ("duration_ms", pa.float32()),     # Nullable
        ("angle_deg", pa.float32()),       # Nullable
        ("distance_px", pa.float32()),     # Nullable
        ("velocity_px_s", pa.float32())    # Nullable
    ])
else:
    CANONICAL_EVENT_SCHEMA = None


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
        c_p = Path(custom_raw_dir).resolve()
        candidates.extend([
            c_p,
            c_p / "adserp-2025",
            c_p / "raw" / "adserp-2025",
            c_p / "adserp",
        ])

    # Priority local paths
    candidates.extend([
        root / ".data" / "raw" / "adserp-2025",
        root / ".data" / "raw" / "raw" / "adserp-2025",
        root / ".data" / "raw" / "adserp",
        root / ".data" / "adserp-2025",
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


def resolve_continuous_kinematics_dir(custom_raw_dir: Optional[str] = None) -> Optional[Path]:
    """Search candidate directories for verified Continuous Kinematics 2020 dataset."""
    root = find_project_root()
    candidates: List[Path] = []
    if custom_raw_dir:
        c_p = Path(custom_raw_dir).resolve()
        candidates.extend([
            c_p,
            c_p / "continuous-kinematics-2020",
            c_p / "raw" / "continuous-kinematics-2020"
        ])
    candidates.extend([
        root / ".data" / "raw" / "continuous-kinematics-2020",
        root / ".data" / "raw" / "raw" / "continuous-kinematics-2020",
        root.parent / "data-sources" / "continuous-kinematics-2020",
        Path("/Users/user/Workspace/MivaCS/FYP/data-sources/continuous-kinematics-2020")
    ])
    for c in candidates:
        if c.is_dir() and (c / "logs").is_dir():
            csv_check = any((c / "logs").glob("*.csv"))
            if csv_check:
                return c
    return None


def resolve_high_volume_trajectories_dir(custom_raw_dir: Optional[str] = None) -> Optional[Path]:
    """Search candidate directories for verified High-Volume Trajectories 20226 dataset."""
    root = find_project_root()
    candidates: List[Path] = []
    if custom_raw_dir:
        c_p = Path(custom_raw_dir).resolve()
        candidates.extend([
            c_p,
            c_p / "high-volume-trajectories-20226",
            c_p / "raw" / "high-volume-trajectories-20226"
        ])
    candidates.extend([
        root / ".data" / "raw" / "high-volume-trajectories-20226",
        root / ".data" / "raw" / "raw" / "high-volume-trajectories-20226",
        root.parent / "data-sources" / "high-volume-trajectories-20226",
        Path("/Users/user/Workspace/MivaCS/FYP/data-sources/high-volume-trajectories-20226")
    ])
    for c in candidates:
        if c.is_dir() and ((c / "dataset.csv").is_file() or (c / "dataset_part.csv").is_file()):
            return c
    return None


def unpack_parquet_sessions(
    parquet_path: Any,
    target_raw_adserp: Any,
    max_sessions: Optional[int] = None
) -> int:
    """
    Unpack consolidated Parquet sessions into individual CSV and XML trial metadata files.
    Restores the standard mouse-movement-data/ and trial-metadata/ hierarchy in milliseconds.
    """
    try:
        import pandas as pd
    except ImportError:
        return 0

    dest = Path(target_raw_adserp)
    mouse_dir = dest / "mouse-movement-data"
    meta_dir = dest / "trial-metadata"
    mouse_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(str(parquet_path))
    session_groups = list(df.groupby("session_id"))
    if max_sessions:
        session_groups = session_groups[:max_sessions]

    count = 0
    for session_id, grp in session_groups:
        csv_p = mouse_dir / f"{session_id}.csv"
        xml_p = meta_dir / f"{session_id}.xml"

        cols = ["timestamp", "xpos", "ypos", "event", "xpath"]
        grp[cols].to_csv(csv_p, index=False)

        r0 = grp.iloc[0]
        xml_content = (
            f"<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n"
            f"<data>\n"
            f" <screen>{int(r0['viewport_w'])}x{int(r0['viewport_h'])}</screen>\n"
            f" <window>{int(r0['viewport_w'])}x{int(r0['viewport_h'])}</window>\n"
            f" <document>{int(r0['doc_w'])}x{int(r0['doc_h'])}</document>\n"
            f"</data>\n"
        )
        with open(xml_p, "w", encoding="utf-8") as f:
            f.write(xml_content)
        count += 1

    print(f"[Data] Unpacked {count} sessions from Parquet into {dest}")
    return count


def ensure_adserp_dataset(
    raw_dir: Optional[str] = None,
    link_to_raw: bool = True
) -> str:
    """
    Ensure the AdSERP dataset is accessible locally.
    Resolves local storage first, then checks local interim Parquet, DVC remote,
    and hosted Hugging Face Hub storage. Fails visibly if data cannot be retrieved.
    """
    root = find_project_root()
    target_raw = Path(raw_dir).resolve() if raw_dir else (root / ".data" / "raw")
    target_raw.mkdir(parents=True, exist_ok=True)
    target_adserp = target_raw / "adserp-2025"

    # 1. Check if already directly present in target raw dir
    if target_adserp.is_dir() and (target_adserp / "mouse-movement-data").is_dir():
        if any((target_adserp / "mouse-movement-data").glob("*.csv")):
            return str(target_adserp)

    # 1b. Check if already present in nested download directory (e.g. .data/raw/raw/adserp-2025)
    nested_adserp = target_raw / "raw" / "adserp-2025"
    if nested_adserp.is_dir() and (nested_adserp / "mouse-movement-data").is_dir():
        if any((nested_adserp / "mouse-movement-data").glob("*.csv")):
            if not target_adserp.exists():
                try:
                    target_adserp.symlink_to(nested_adserp, target_is_directory=True)
                    print(f"[Data] Linked nested download: {target_adserp} -> {nested_adserp}")
                except Exception:
                    # If symlinking fails on host, move it directly into place
                    try:
                        shutil.move(str(nested_adserp), str(target_adserp))
                        print(f"[Data] Moved nested download to canonical location: {target_adserp}")
                    except Exception:
                        return str(nested_adserp)
            return str(target_adserp)

    # 2. Check candidate local paths (e.g. data-sources/adserp-2025)
    found = resolve_adserp_dir(raw_dir)
    if found:
        if link_to_raw and not target_adserp.exists():
            try:
                target_adserp.symlink_to(found, target_is_directory=True)
                print(f"[Data] Created inode-safe symlink: {target_adserp} -> {found}")
                return str(target_adserp)
            except Exception:
                return str(found)
        return str(found)

    # 3. Check if cached consolidated Parquet exists locally in .data/interim/
    interim_dir = root / ".data" / "interim"
    parquet_local = interim_dir / "adserp_consolidated.parquet"
    if parquet_local.is_file():
        print(f"[Data] Unpacking sessions from local consolidated archive: {parquet_local}")
        unpacked = unpack_parquet_sessions(parquet_local, target_adserp)
        if unpacked > 0:
            return str(target_adserp)

    # 4. Try DVC pull
    print("[Data] AdSERP not found locally; querying DVC remote...")
    if pull_dvc_dataset("adserp-2025"):
        found = resolve_adserp_dir(raw_dir)
        if found:
            return str(found)

    # 5. Remote Hugging Face Hub download (fast consolidated Parquet retrieval)
    print("[Data] Downloading consolidated AdSERP archive from Hugging Face Hub (T40/edge-aui-framework-data)...")
    try:
        from huggingface_hub import hf_hub_download
        interim_dir.mkdir(parents=True, exist_ok=True)
        downloaded_parquet = hf_hub_download(
            repo_id="T40/edge-aui-framework-data",
            repo_type="dataset",
            filename="interim/adserp_consolidated.parquet",
            local_dir=str(root / ".data")
        )
        print(f"[Data] Successfully retrieved remote consolidated archive: {downloaded_parquet}")
        unpacked = unpack_parquet_sessions(downloaded_parquet, target_adserp)
        if unpacked > 0:
            return str(target_adserp)
    except Exception as e:
        print(f"[Data] Note on remote archive sync: {e}")

    # 6. Fallback: On-demand download of sample sessions from HF Hub
    try:
        from huggingface_hub import hf_hub_download
        mouse_dir = target_adserp / "mouse-movement-data"
        meta_dir = target_adserp / "trial-metadata"
        mouse_dir.mkdir(parents=True, exist_ok=True)
        meta_dir.mkdir(parents=True, exist_ok=True)

        for s_id in ["p004-b1-t1", "p004-b1-t2", "p004-b1-t3", "p004-b1-t4", "p004-b1-t5"]:
            csv_path = hf_hub_download(
                repo_id="T40/edge-aui-framework-data",
                repo_type="dataset",
                filename=f"raw/adserp-2025/mouse-movement-data/{s_id}.csv"
            )
            xml_path = hf_hub_download(
                repo_id="T40/edge-aui-framework-data",
                repo_type="dataset",
                filename=f"raw/adserp-2025/trial-metadata/{s_id}.xml"
            )
            shutil.copy2(csv_path, mouse_dir / f"{s_id}.csv")
            shutil.copy2(xml_path, meta_dir / f"{s_id}.xml")

        if any(mouse_dir.glob("*.csv")):
            return str(target_adserp)
    except Exception as e:
        print(f"[Data] Note on direct session retrieval: {e}")

    # 7. Fail visibly (no manufactured synthetic fallback)
    raise FileNotFoundError(
        f"AdSERP dataset could not be located or retrieved. "
        f"Checked local directories ({target_adserp}, data-sources/adserp-2025) and remote Hugging Face Hub storage. "
        f"Please verify data sources or network connectivity."
    )


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


# ============================================================================
# Canonical Parquet Conversion Adapters (Phase B, C, E, F)
# ============================================================================

def convert_adserp_to_canonical(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_sessions: Optional[int] = None,
    force: bool = False
) -> str:
    """
    Convert raw AdSERP CSV and companion XML files into canonical Parquet format.
    Stores canonical events under .data/canonical/adserp/data.parquet.
    """
    if not PYARROW_AVAILABLE or CANONICAL_EVENT_SCHEMA is None:
        raise RuntimeError("PyArrow is required for canonical Parquet conversion.")

    root = find_project_root()
    adserp_dir = Path(raw_path or resolve_adserp_dir() or ensure_adserp_dataset())
    if not adserp_dir or not adserp_dir.is_dir():
        raise FileNotFoundError(f"AdSERP dataset directory not found at {adserp_dir}")

    out_file = Path(output_path or (root / ".data" / "canonical" / "adserp" / "data.parquet")).resolve()
    if not force and out_file.is_file() and out_file.stat().st_size > 0:
        print(f"[Canonical] Using cached canonical AdSERP Parquet: {out_file}")
        return str(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    mouse_dir = adserp_dir / "mouse-movement-data"
    meta_dir = adserp_dir / "trial-metadata"

    if not mouse_dir.is_dir():
        raise FileNotFoundError(f"Missing mouse-movement-data directory at {mouse_dir}")

    csv_files = sorted(list(mouse_dir.glob("*.csv")))
    if max_sessions:
        csv_files = csv_files[:max_sessions]

    print(f"[Canonical] Converting {len(csv_files)} AdSERP sessions to canonical Parquet ({out_file})...")

    writer = pq.ParquetWriter(str(out_file), schema=CANONICAL_EVENT_SCHEMA, compression="snappy")
    batch_rows: List[Dict[str, Any]] = []
    total_events = 0

    try:
        for i, csv_file in enumerate(csv_files):
            session_id = csv_file.stem
            parts = session_id.split("-")
            user_id = parts[0] if len(parts) >= 1 else None
            trial_id = "-".join(parts[1:]) if len(parts) > 1 else None

            xml_file = meta_dir / f"{session_id}.xml"
            if not xml_file.is_file():
                continue
            try:
                (vp_w, vp_h), (doc_w, doc_h) = parse_viewport_metadata(str(xml_file))
            except Exception:
                continue

            session_events = []
            with open(csv_file, "r", encoding="utf-8") as f:
                header = f.readline().strip().split(",")
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    p = line_str.split(",")
                    if len(p) >= 4:
                        try:
                            ts = int(p[0])
                            x = float(p[1])
                            y = float(p[2])
                            ev = p[3]
                            xpath = p[4] if len(p) > 4 and p[4] not in ("", "/", "/html") else None

                            x_norm = max(0.0, min(1.0, x / max(vp_w, 1.0)))
                            y_norm = max(0.0, min(1.0, y / max(vp_h, 1.0)))

                            session_events.append({
                                "dataset_id": "adserp",
                                "session_id": session_id,
                                "user_id": user_id,
                                "trial_id": trial_id,
                                "timestamp_ms": ts,
                                "event_type": ev,
                                "x": float(x),
                                "y": float(y),
                                "x_norm": float(x_norm),
                                "y_norm": float(y_norm),
                                "viewport_width": float(vp_w),
                                "viewport_height": float(vp_h),
                                "document_width": float(doc_w),
                                "document_height": float(doc_h),
                                "target_id": xpath,
                                "source_event_id": None,
                                "duration_ms": None,
                                "angle_deg": None,
                                "distance_px": None,
                                "velocity_px_s": None
                            })
                        except (ValueError, IndexError):
                            continue

            session_events.sort(key=lambda e: e["timestamp_ms"])
            batch_rows.extend(session_events)

            if len(batch_rows) >= 50_000 or (i == len(csv_files) - 1 and batch_rows):
                table = pa.Table.from_pylist(batch_rows, schema=CANONICAL_EVENT_SCHEMA)
                writer.write_table(table)
                total_events += len(batch_rows)
                batch_rows = []

    finally:
        writer.close()

    print(f"[Canonical] Successfully wrote {total_events} events across {len(csv_files)} sessions to {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(out_file)


def convert_continuous_kinematics_to_canonical(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_sessions: Optional[int] = None,
    force: bool = False
) -> str:
    """
    Convert Continuous Kinematics 2020 space-delimited logs and companion XMLs into canonical Parquet.
    Joins participants.tsv to assign verified user_id to each session.
    """
    if not PYARROW_AVAILABLE or CANONICAL_EVENT_SCHEMA is None:
        raise RuntimeError("PyArrow is required for canonical Parquet conversion.")

    root = find_project_root()
    ck_dir = Path(raw_path or resolve_continuous_kinematics_dir())
    if not ck_dir or not ck_dir.is_dir():
        raise FileNotFoundError(f"Continuous Kinematics dataset directory not found at {ck_dir}")

    out_file = Path(output_path or (root / ".data" / "canonical" / "continuous_kinematics" / "data.parquet")).resolve()
    if not force and out_file.is_file() and out_file.stat().st_size > 0:
        print(f"[Canonical] Using cached canonical Continuous Kinematics Parquet: {out_file}")
        return str(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)

    # Load participants.tsv mapping log_id -> (user_id, serp_id)
    participants_map: Dict[str, Tuple[str, str]] = {}
    parts_tsv = ck_dir / "participants.tsv"
    if parts_tsv.is_file():
        with open(parts_tsv, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            user_idx = header.index("user_id") if "user_id" in header else 0
            log_idx = header.index("log_id") if "log_id" in header else 11
            serp_idx = header.index("serp_id") if "serp_id" in header else 9
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) > max(user_idx, log_idx, serp_idx):
                    participants_map[parts[log_idx]] = (parts[user_idx], parts[serp_idx])

    logs_dir = ck_dir / "logs"
    if not logs_dir.is_dir():
        raise FileNotFoundError(f"Missing logs directory in Continuous Kinematics at {logs_dir}")

    csv_files = sorted(list(logs_dir.glob("*.csv")))
    if max_sessions:
        csv_files = csv_files[:max_sessions]

    print(f"[Canonical] Converting {len(csv_files)} Continuous Kinematics sessions to canonical Parquet ({out_file})...")

    writer = pq.ParquetWriter(str(out_file), schema=CANONICAL_EVENT_SCHEMA, compression="snappy")
    batch_rows: List[Dict[str, Any]] = []
    total_events = 0

    try:
        for i, csv_file in enumerate(csv_files):
            log_id = csv_file.stem
            user_id, trial_id = participants_map.get(log_id, (None, None))

            xml_file = logs_dir / f"{log_id}.xml"
            vp_w, vp_h = 1366.0, 768.0
            doc_w, doc_h = 1366.0, 2000.0
            if xml_file.is_file():
                try:
                    (vp_w, vp_h), (doc_w, doc_h) = parse_viewport_metadata(str(xml_file))
                except Exception:
                    pass

            session_events = []
            with open(csv_file, "r", encoding="utf-8") as f:
                header = f.readline().strip().split()
                for line in f:
                    line_str = line.strip()
                    if not line_str:
                        continue
                    p = line_str.split()
                    if len(p) >= 5:
                        try:
                            ts = int(p[1])
                            x = float(p[2])
                            y = float(p[3])
                            ev = p[4]
                            xpath = p[5] if len(p) > 5 and p[5] not in ("/", "/html", "{}") else None

                            x_norm = max(0.0, min(1.0, x / max(vp_w, 1.0)))
                            y_norm = max(0.0, min(1.0, y / max(vp_h, 1.0)))

                            session_events.append({
                                "dataset_id": "continuous_kinematics",
                                "session_id": log_id,
                                "user_id": user_id,
                                "trial_id": trial_id,
                                "timestamp_ms": ts,
                                "event_type": ev,
                                "x": float(x),
                                "y": float(y),
                                "x_norm": float(x_norm),
                                "y_norm": float(y_norm),
                                "viewport_width": float(vp_w),
                                "viewport_height": float(vp_h),
                                "document_width": float(doc_w),
                                "document_height": float(doc_h),
                                "target_id": xpath,
                                "source_event_id": p[0],
                                "duration_ms": None,
                                "angle_deg": None,
                                "distance_px": None,
                                "velocity_px_s": None
                            })
                        except (ValueError, IndexError):
                            continue

            session_events.sort(key=lambda e: e["timestamp_ms"])
            batch_rows.extend(session_events)

            if len(batch_rows) >= 50_000 or (i == len(csv_files) - 1 and batch_rows):
                table = pa.Table.from_pylist(batch_rows, schema=CANONICAL_EVENT_SCHEMA)
                writer.write_table(table)
                total_events += len(batch_rows)
                batch_rows = []

    finally:
        writer.close()

    print(f"[Canonical] Successfully wrote {total_events} events across {len(csv_files)} sessions to {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(out_file)


def convert_high_volume_trajectories_to_canonical(
    raw_path: Optional[str] = None,
    output_path: Optional[str] = None,
    max_rows: Optional[int] = None,
    chunk_size: int = 500_000,
    force: bool = False
) -> str:
    """
    Convert High-Volume Trajectories 20226 semicolon-delimited CSV into canonical Parquet.
    Streams large files in chunks without fabricating web coordinates or viewports.
    """
    if not PYARROW_AVAILABLE or CANONICAL_EVENT_SCHEMA is None:
        raise RuntimeError("PyArrow is required for canonical Parquet conversion.")

    try:
        import pandas as pd
    except ImportError:
        raise RuntimeError("pandas is required for streaming high-volume trajectory chunks.")

    root = find_project_root()
    hvt_dir = Path(raw_path or resolve_high_volume_trajectories_dir())
    if not hvt_dir or not hvt_dir.is_dir():
        raise FileNotFoundError(f"High-Volume Trajectories dataset directory not found at {hvt_dir}")

    csv_candidate = hvt_dir / "dataset.csv"
    if not csv_candidate.is_file():
        csv_candidate = hvt_dir / "dataset_part.csv"
    if not csv_candidate.is_file():
        raise FileNotFoundError(f"Neither dataset.csv nor dataset_part.csv found in {hvt_dir}")

    out_file = Path(output_path or (root / ".data" / "canonical" / "high_volume_trajectories" / "data.parquet")).resolve()
    if not force and out_file.is_file() and out_file.stat().st_size > 0:
        print(f"[Canonical] Using cached canonical High-Volume Trajectories Parquet: {out_file}")
        return str(out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Canonical] Streaming {csv_candidate} in chunks of {chunk_size} rows to canonical Parquet ({out_file})...")

    writer = pq.ParquetWriter(str(out_file), schema=CANONICAL_EVENT_SCHEMA, compression="snappy")
    total_events = 0
    reader = pd.read_csv(csv_candidate, sep=";", chunksize=chunk_size, low_memory=False)

    try:
        for chunk in reader:
            if max_rows and total_events >= max_rows:
                break
            if max_rows and total_events + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - total_events]

            # Timestamp parsing
            dt_series = pd.to_datetime(chunk["local_date"].astype(str) + " " + chunk["local_time"].astype(str), errors="coerce")
            ts_ms = (dt_series.astype("int64") // 10**6).values

            user_ids = chunk["idman"].astype(str).values
            durations = pd.to_numeric(chunk["duration"], errors="coerce").values
            angles = pd.to_numeric(chunk["angle"], errors="coerce").values
            distances = pd.to_numeric(chunk["distance"], errors="coerce").values
            velocities = pd.to_numeric(chunk["velocity"], errors="coerce").values

            # event_type: dwell or movement
            event_types = ["dwell" if (d is not None and not (isinstance(d, float) and pd.isna(d)) and d > 0) else "movement" for d in durations]

            batch_dict = {
                "dataset_id": ["high_volume_trajectories"] * len(chunk),
                "session_id": user_ids.tolist(),
                "user_id": user_ids.tolist(),
                "trial_id": [None] * len(chunk),
                "timestamp_ms": ts_ms.tolist(),
                "event_type": event_types,
                "x": [None] * len(chunk),
                "y": [None] * len(chunk),
                "x_norm": [None] * len(chunk),
                "y_norm": [None] * len(chunk),
                "viewport_width": [None] * len(chunk),
                "viewport_height": [None] * len(chunk),
                "document_width": [None] * len(chunk),
                "document_height": [None] * len(chunk),
                "target_id": [None] * len(chunk),
                "source_event_id": [None] * len(chunk),
                "duration_ms": [float(d) if pd.notna(d) else None for d in durations],
                "angle_deg": [float(a) if pd.notna(a) else None for a in angles],
                "distance_px": [float(dist) if pd.notna(dist) else None for dist in distances],
                "velocity_px_s": [float(v) if pd.notna(v) else None for v in velocities]
            }

            table = pa.Table.from_pydict(batch_dict, schema=CANONICAL_EVENT_SCHEMA)
            writer.write_table(table)
            total_events += len(chunk)
            print(f"  Processed {total_events:,} events...")

    finally:
        reader.close()
        writer.close()

    print(f"[Canonical] Successfully wrote {total_events:,} events to {out_file} ({out_file.stat().st_size / 1024 / 1024:.2f} MB)")
    return str(out_file)


def canonicalize_all_datasets(
    canonical_dir: Optional[str] = None,
    force: bool = False
) -> Dict[str, str]:
    """
    Run canonical conversion across all candidate interaction datasets.
    Creates canonical Parquet files under .data/canonical/.
    """
    root = find_project_root()
    base_canon = Path(canonical_dir or (root / ".data" / "canonical")).resolve()
    results: Dict[str, str] = {}

    # 1. AdSERP
    try:
        adserp_out = base_canon / "adserp" / "data.parquet"
        results["adserp"] = convert_adserp_to_canonical(output_path=str(adserp_out), force=force)
    except Exception as e:
        print(f"[Canonical] Note on AdSERP conversion: {e}")

    # 2. Continuous Kinematics
    try:
        ck_out = base_canon / "continuous_kinematics" / "data.parquet"
        results["continuous_kinematics"] = convert_continuous_kinematics_to_canonical(output_path=str(ck_out), force=force)
    except Exception as e:
        print(f"[Canonical] Note on Continuous Kinematics conversion: {e}")

    # 3. High-Volume Trajectories
    try:
        hvt_out = base_canon / "high_volume_trajectories" / "data.parquet"
        results["high_volume_trajectories"] = convert_high_volume_trajectories_to_canonical(output_path=str(hvt_out), force=force)
    except Exception as e:
        print(f"[Canonical] Note on High-Volume Trajectories conversion: {e}")

    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Edge-AUI Data Ingestion & Canonical Parquet Driver")
    parser.add_argument("--mode", type=str, default="minimal")
    parser.add_argument("--max-sessions", type=int, default=None, help="Cap sessions for test run")
    parser.add_argument("--max-rows", type=int, default=None, help="Cap rows for HVT test run")
    parser.add_argument("--canonicalize", type=str, default=None, choices=["all", "adserp", "ck", "hvt"], help="Execute canonical Parquet conversion")
    parser.add_argument("--force", action="store_true", help="Force re-conversion")
    args = parser.parse_args()

    if args.canonicalize:
        if args.canonicalize == "all":
            canonicalize_all_datasets(force=args.force)
        elif args.canonicalize == "adserp":
            convert_adserp_to_canonical(max_sessions=args.max_sessions, force=args.force)
        elif args.canonicalize == "ck":
            convert_continuous_kinematics_to_canonical(max_sessions=args.max_sessions, force=args.force)
        elif args.canonicalize == "hvt":
            convert_high_volume_trajectories_to_canonical(max_rows=args.max_rows, force=args.force)
    else:
        adserp_path = ensure_adserp_dataset()
        print(f"[Data Driver] Verified AdSERP path: {adserp_path}")
        consolidated = consolidate_adserp_sessions(raw_path=adserp_path, max_sessions=args.max_sessions or 50, force=args.force)
        print(f"[Data Driver] Consolidated dataset ready at: {consolidated}")
